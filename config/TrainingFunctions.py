import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.optim as optim
import pickle
from time import sleep
from config.Constants import *
from config.LoadData import load_trained_module
from config.PlotFile import plot
from config.Operators import D_WC
from config.TransformerModules import PTC
from config.GMRES import gmres, gmres_precon, gmres_train


# the loss function defines how far the module output is off the desired target (it shall be minimized)
def loss_fn(module_output, target):
    diff = (module_output - target).view(-1)
    return torch.norm(diff) / target.shape[0] ** (1 / 2)


# in the training function there will be the option to use gmres for target generation, there a
# preconditioner can be deployed to cut calculation time
best_preconditioner = load_trained_module(PTC, "Ptc_4_4")
for param in best_preconditioner.parameters():
    param.requires_grad = False
best_preconditioner_small = load_trained_module(PTC, "ptc_4_4_gmres")
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
        self.optimizer = optim.Adam(self.module.parameters(), lr=0.01)
        self.structure = structure
        self.epoch_list = []
        self.loss_list = []
        self.fig, self.ax = plt.subplots()

    def save_data_as(self, save_model_name: str):
        self.save_plot(save_model_name)
        self.save_structure(save_model_name)
        self.save_parameters(save_model_name)
        print("Data successfully saved!")

    def save_parameters(self, file_name):
        torch.save(self.module.state_dict(), "config/Saved_paras/" + file_name + ".pth")

    def save_plot(self, file_name):
        with open("config/Saved_plots/" + file_name + ".txt", "w") as file:
            for epoch, loss in zip(self.epoch_list, self.loss_list):
                file.write(f"{epoch}\t{loss}\n")

    def save_structure(self, file_name):
        with open("config/Saved_structures/" + file_name + ".pkl", 'wb') as f:
            pickle.dump(self.structure, f)

    # Can be stopped and resumed
    def interactive_training(self, small=False, train_with_gmres=False, sample_update_period=1, hard_stop=4000):
        global best_preconditioner, best_preconditioner_small

        if small:
            opterator = D_WC(M, GAUGE_FIELD_SMALL)
            preconditioner = best_preconditioner_small
            lattice = LATTICE_SMALL
        else:
            opterator = D_WC(M, GAUGE_FIELD)
            preconditioner = best_preconditioner
            lattice = LATTICE
        if train_with_gmres:
            assert DEFAULT_BATCH_SIZE == 1, "Default Batch size must be 1 to training with GMRES!"
        else:
            preconditioner = None

        try:
            # Start conditions
            counter = len(self.epoch_list)
            current_B, target = self.new_sample(lattice, opterator, preconditioner)
            lowest_mean = 1e6
            stop_counter = 0
            plot("Update Plot", self.epoch_list, self.loss_list)
            converged = False
            while not converged:
                if counter % sample_update_period == 0:
                    current_B, target = self.new_sample(lattice, opterator, preconditioner)
                counter += 1
                self.training_step(current_B, target)
                self.epoch_list.append(counter)
                # Sleep to protect CPU
                sleep(0.1)

                if (counter - 1) % 10 == 0:
                    clear_output(wait=True)
                    self.plot_data()
                    converged, lowest_mean = self.check_if_converged(counter, lowest_mean, stop_counter)
                    if counter > hard_stop:
                        break

            clear_output(wait=True)
            plot("Update Plot", self.epoch_list, self.loss_list)
            print("\n" + "Goal reached!")

        except KeyboardInterrupt:
            clear_output(wait=True)
            plot("Update Plot", self.epoch_list, self.loss_list)
            print("\n" + "Training stopped manually!")

    # intended to be used with untrained model
    def scripted_training(self, small=False):
        assert DEFAULT_BATCH_SIZE == 1, "Default Batch size must be 1 to training with GMRES!"
        if small:
            opterator = D_WC(M, GAUGE_FIELD_SMALL)
            lattice = LATTICE_SMALL
        else:
            opterator = D_WC(M, GAUGE_FIELD)
            lattice = LATTICE

        # First train the model without gmres:
        self.epoch_list = []
        self.loss_list = []
        counter = 0
        lowest_mean = 1e6
        stop_counter = 0
        converged = False
        while not converged:
            current_B, target = self.new_sample(lattice, opterator, None)
            counter += 1
            self.training_step(current_B, target)
            self.epoch_list.append(counter)
            # Sleep to protect CPU
            sleep(0.1)
            if (counter - 1) % 10 == 0:
                converged, lowest_mean = self.check_if_converged(counter, lowest_mean, stop_counter)

        # Resume training with gmres and the now trained model itself as preconditioner
        stop_counter = 0
        while True:
            current_B, target = self.new_sample(lattice, opterator, self.module)
            counter += 1
            self.training_step(current_B, target)
            self.epoch_list.append(counter)
            # Sleep to protect CPU
            sleep(0.1)
            if (counter - 1) % 10 == 0:
                converged, lowest_mean = self.check_if_converged(counter, lowest_mean, stop_counter)

    def training_step(self, current_B, target):
        # defining function for parameters to minimize
        loss = loss_fn(self.module(current_B), target)
        # calculating parameter gradients
        loss.backward()
        # changing parameters
        self.optimizer.step()
        # zeroing gradients
        self.optimizer.zero_grad()
        self.loss_list.append(loss.item())

    def new_sample(self, lattice, opterator, preconditioner):
        with torch.no_grad():
            rand_field = torch.rand(DEFAULT_BATCH_SIZE, *lattice, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
            if preconditioner is not None:
                target = gmres_precon(opterator, rand_field, preconditioner)[0]
                current_B = rand_field
            else:
                target = rand_field
                current_B = opterator(rand_field)
        return current_B, target

    def check_if_converged(self, counter, lowest_mean, stop_counter):
        if counter > 100:
            current_mean = torch.mean(torch.tensor(self.loss_list[-100:]))
            if lowest_mean - 1e-2 < current_mean:
                stop_counter += 1
            else:
                stop_counter = 0
                lowest_mean = current_mean.clone()
            if stop_counter == 10:
                return True, lowest_mean
        return False, lowest_mean
