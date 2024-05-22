import matplotlib.pyplot as plt
import torch.optim as optim
import pickle
from time import sleep
from modules.Preconditioner import ptc
from modules.Operators import D_WC
from modules.Constants import *
from modules.GMRES import gmres


# the loss function defines how far the module output is off the desired target (it shall be minimized)
def loss_fn(module_output, target):
    diff = (module_output - target).view(-1)
    return torch.norm(diff) / target.shape[0] ** (1 / 2)


# in the training function there will be the option to use gmres for target generation, there a
# preconditioner can be deployed to cut calculation time
best_preconditioner = ptc("Ptc_4_4")
for param in best_preconditioner.parameters():
    param.requires_grad = False
best_preconditioner_small = ptc("ptc_gmres_4_4")
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
    def __init__(self, module, structure=None):
        global best_preconditioner

        self.module = module
        self.structure = structure
        self.structure_name = ""
        if structure is not None:
            for info in self.structure:
                if isinstance(info, int):
                    self.structure_name += "_" + str(info)

        self.epoch_list = []
        self.loss_list = []

    def safe_data(self, file_name="Tr"):
        file_name += self.structure_name

        with open("modules/Saved_plots/" + file_name + ".txt", "w") as file:
            for epoch, loss in zip(self.epoch_list, self.loss_list):
                file.write(f"{epoch}\t{loss}\n")

        with open("modules/Saved_structures/" + file_name + ".pkl", 'wb') as f:
            pickle.dump(self.structure, f)

        torch.save(self.module.state_dict(), "modules/Saved_paras/" + file_name + ".pth")
        print("Data were successfully saved!")

    def plot_data(self):
        plot_name = self.structure_name
        plt.plot(self.epoch_list, self.loss_list, marker='o', linestyle='-', markersize=0.1)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(plot_name)
        plt.grid(True)
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
        plt.show()

    def train(self, small=True, train_with_gmres=False, learning_rate=0.01, batch_size=1, sample_training_length=1, feedback_timer=10, update_plot=False, stop_condition=1e-2,
              stop_condition_length=100, hard_stop=4000):
        global best_preconditioner, best_preconditioner_small
        optimizer = optim.Adam(self.module.parameters(), lr=learning_rate)
        counter = len(self.epoch_list)
        if small:
            opterator = D_WC(M, GAUGE_FIELD_SMALL)
            preconditioner = best_preconditioner_small
            lattice = LATTICE_SMALL
        else:
            opterator = D_WC(M, GAUGE_FIELD)
            preconditioner = best_preconditioner
            lattice = LATTICE

        if train_with_gmres:
            assert batch_size == 1, "Batch size must be 1 for training with GMRES!"

        # first training sample
        with torch.no_grad():
            if train_with_gmres:
                current_B = torch.rand(batch_size, *lattice, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
                target = gmres(opterator, current_B, preconditioner, preconditioner(current_B))[0]
            else:
                target = torch.rand(batch_size, *lattice, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
                current_B = opterator(target)

        # Start_values for stop condition
        lowest_mean = 1e6
        stop_counter = 0
        eps = stop_condition
        N = stop_condition_length

        try:
            if update_plot:
                self.plot_data()
            while True:
                # Creating new sample each n-th round
                if counter % sample_training_length == 0:
                    with torch.no_grad():
                        if train_with_gmres:
                            current_B = torch.rand(batch_size, *lattice, GAUGE_DOF, NON_GAUGE_DOF,
                                                   dtype=torch.complex64)
                            target = gmres(opterator, current_B, preconditioner, preconditioner(current_B))[0]
                        else:
                            target = torch.rand(batch_size, *lattice, GAUGE_DOF, NON_GAUGE_DOF,
                                                dtype=torch.complex64)
                            current_B = opterator(target)

                # zeroing gradients
                optimizer.zero_grad()

                # Calculating the loss
                loss = loss_fn(self.module(current_B), target)

                # Backward pass
                loss.backward()

                # changing parameters
                optimizer.step()

                counter += 1

                # Appending epoch and loss values to lists
                self.epoch_list.append(counter)
                self.loss_list.append(loss.item())
                if (counter - 1) % feedback_timer == 0:
                    if update_plot:
                        self.plot_data()

                    # If the mean of the last N loss_list values does not decrease after N times in succession
                    if counter > N:
                        current_mean = torch.mean(torch.tensor(self.loss_list[-N:]))
                        if lowest_mean - eps < current_mean:
                            stop_counter += 1
                        else:
                            stop_counter = 0
                            lowest_mean = current_mean.clone()
                        if stop_counter == N // feedback_timer:
                            break

                    # Hard Stop Condition
                    if counter > hard_stop:
                        break

                # Sleep to protect cpu
                sleep(0.1)

            if update_plot:
                self.plot_data()
            print("\n" + "Goal reached!")

        except KeyboardInterrupt:
            if update_plot:
                self.plot_data()
            print("\n" + "Training stopped manually!")
