import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.optim as optim
import pickle
from time import sleep
from configuration.LoadData import load_trained_module
from configuration.Operators import D_WC
from configuration.TransformerModules import PTC
from configuration.Constants import *
from configuration.GMRES import gmres


# the loss function defines how far the module output is off the desired target (it shall be minimized)
def loss_fn(module_output, target):
    diff = (module_output - target).view(-1)
    return torch.norm(diff) / target.shape[0] ** (1 / 2)


# in the training function there will be the option to use gmres for target generation, there a
# preconditioner can be deployed to cut calculation time
best_preconditioner = load_trained_module(PTC, "Ptc_4_4")
for param in best_preconditioner.parameters():
    param.requires_grad = False
best_preconditioner_small = load_trained_module(PTC, "ptc_gmres_4_4")
for param in best_preconditioner_small.parameters():
    param.requires_grad = False


"""""
The Class DwcTrainer is a class designed for training a nn.Module to become a good preconditioner for solving Dwc(x) = b
with GMRES.
To achieve this the Module should approximate the inverse of Dwc by minimizing the loss function in two possible ways.
One way is to calculate b = Dwc(x) for random x and then use Module on b to revert it back to x.
The other way is to use GMRES on a random b to calculate the exact x_gmres (with small error) an then to use the Module also 
on b to approximate the GMRES solution x_gmres with it.
Further there is the option to train on large (8^3 x 16) or small (4^3 x 8) Lattice.
In addition DwcTrainer has the option to save all important data of the Training (Module parameters and structure and the
training plot-data.
"""""


class DwcTrainer:
    def __init__(self, module, structure):
        global best_preconditioner

        self.module = module
        self.structure = structure
        self.epoch_list = []
        self.loss_list = []
        self.fig, self.ax = plt.subplots()

    def safe_data_as(self, save_model_name: str):
        self.save_plot(save_model_name)
        self.save_structure(save_model_name)
        self.save_parameters(save_model_name)
        print("Data were successfully saved!")

    def plot_data(self):
        self.set_plot_base()
        self.add_mean_and_var()
        plt.show()

    def train(self, small=False, train_with_gmres=False, learning_rate=0.01, batch_size=1, sample_update_period=1,
              check_period=10, update_plot=False, min_mean_diff=1e-2, stop_repetition_length=100, max_epoch=4000):

        optimizer = optim.Adam(self.module.parameters(), lr=learning_rate)
        counter = len(self.epoch_list)
        lattice, opterator, preconditioner = self.init_training_constants(small)

        if train_with_gmres:
            assert batch_size == 1, "Batch size must be 1 for training with GMRES!"

        # first training sample
        current_B, target = self.new_sample(batch_size, lattice, opterator, preconditioner, train_with_gmres)

        # Start_values for stop condition
        lowest_mean = 1e6
        stop_counter = 0

        try:
            if update_plot:
                self.plot_data()
            running = True
            while running:
                if counter % sample_update_period == 0:
                    current_B, target = self.new_sample(batch_size, lattice, opterator, preconditioner,
                                                        train_with_gmres)
                counter += 1
                self.training_step(counter, current_B, optimizer, target)
                if (counter - 1) % check_period == 0:
                    if update_plot:
                        clear_output(wait=True)
                        self.plot_data()
                    running, lowest_mean, stop_counter = self.check_stop_conditions(counter, check_period, max_epoch, lowest_mean,
                                                                      running, min_mean_diff, stop_repetition_length,
                                                                      stop_counter)

                # Sleep to protect CPU
                sleep(0.1)

            if update_plot:
                clear_output(wait=True)
                self.plot_data()
            print("\n" + "Goal reached!")

        except KeyboardInterrupt:
            if update_plot:
                clear_output(wait=True)
                self.plot_data()
            print("\n" + "Training stopped manually!")

    def save_parameters(self, model_name):
        torch.save(self.module.state_dict(), "configuration/Saved_paras/" + model_name + ".pth")

    def save_plot(self, file_name):
        with open("configuration/Saved_plots/" + file_name + ".txt", "w") as file:
            for epoch, loss in zip(self.epoch_list, self.loss_list):
                file.write(f"{epoch}\t{loss}\n")

    def save_structure(self, file_name):
        with open("configuration/Saved_structures/" + file_name + ".pkl", 'wb') as f:
            pickle.dump(self.structure, f)

    def add_mean_and_var(self):
        if len(self.epoch_list) > 0:
            last_epoch = self.epoch_list[-1]
            if len(self.loss_list) < 50:
                last_loss = self.loss_list[-1]
                plt.annotate(f'Last Epoch: {last_epoch}, Last Loss: {last_loss:.2f}', xy=(last_epoch, last_loss),
                                 xytext=(20, 20),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))
            else:
                loss_tensor = torch.tensor(self.loss_list[-50:])
                loss_mean = torch.mean(loss_tensor).item()
                loss_var = torch.std(loss_tensor).item()
                plt.annotate(f'Last Epoch: {last_epoch}, Mean: {loss_mean:.2f}, Variance: {loss_var:.2f}',
                                 xy=(last_epoch, loss_mean), xytext=(20, 20),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))

    def set_plot_base(self):
        plt.plot(self.epoch_list, self.loss_list, marker='o', linestyle='-', markersize=0.1)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Update Plot")
        plt.grid(True)

    def check_stop_conditions(self, counter, feedback_timer, hard_stop, lowest_mean, running, stop_condition,
                              stop_condition_length,
                              stop_counter):
        if counter > stop_condition_length:
            current_mean = torch.mean(torch.tensor(self.loss_list[-stop_condition_length:]))
            if lowest_mean - stop_condition < current_mean:
                stop_counter += 1
            else:
                stop_counter = 0
                lowest_mean = current_mean.clone()
            if stop_counter == stop_condition_length // feedback_timer:
                running = False
        if counter > hard_stop:
            running = False
        return running, lowest_mean, stop_counter

    def training_step(self, counter, current_B, optimizer, target):
        # zeroing gradients
        optimizer.zero_grad()
        # Calculating the loss
        loss = loss_fn(self.module(current_B), target)
        # Backward pass
        loss.backward()
        # changing parameters
        optimizer.step()
        # Appending epoch and loss values to lists
        self.epoch_list.append(counter)
        self.loss_list.append(loss.item())

    def init_training_constants(self, small):
        global best_preconditioner, best_preconditioner_small
        if small:
            opterator = D_WC(M, GAUGE_FIELD_SMALL)
            preconditioner = best_preconditioner_small
            lattice = LATTICE_SMALL
        else:
            opterator = D_WC(M, GAUGE_FIELD)
            preconditioner = best_preconditioner
            lattice = LATTICE
        return lattice, opterator, preconditioner

    def new_sample(self, batch_size, lattice, opterator, preconditioner, train_with_gmres):
        with torch.no_grad():
            if train_with_gmres:
                current_B = torch.rand(batch_size, *lattice, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
                target = gmres(opterator, current_B, preconditioner, preconditioner(current_B))[0]
            else:
                target = torch.rand(batch_size, *lattice, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
                current_B = opterator(target)
        return current_B, target
