import time
import matplotlib.pyplot as plt
import torch.optim as optim
import pickle
from IPython.display import clear_output
from modules.Preconditioner import ptc
from modules.Constants import *
from modules.Operators import D_WC
from modules.GMRES import gmres


def loss_fn(module_output, target):
    diff = (module_output - target).view(-1)
    return torch.norm(diff) / target.shape[0] ** (1 / 2)


# D_WC
d_wc = D_WC(M, GAUGE_FIELD)

# Preconditioner for faster GMRES
# best_preconditioner = ptc("Ptc_gmres_4_4")
# for param in best_preconditioner.parameters():
#     param.requires_grad = False


class DwcTrainer:
    def __init__(self, module, structure=None, learning_rate=0.01):
        global best_preconditioner

        self.module = module
        self.optimizer = optim.Adam(self.module.parameters(), lr=learning_rate)
        self.structure = structure
        self.lr = learning_rate
        self.structure_name = ""
        if structure is not None:
            for info in self.structure:
                if isinstance(info, int):
                    self.structure_name += "_" + str(info)

        self.epoch_list = []
        self.loss_list = []

    def safe_data(self, file_name="Tr"):
        file_name += self.structure_name

        with open("Saved_plots/" + file_name + ".txt", "w") as file:
            for epoch, loss in zip(self.epoch_list, self.loss_list):
                file.write(f"{epoch}\t{loss}\n")

        with open("Saved_structures/" + file_name + ".pkl", 'wb') as f:
            pickle.dump(self.structure, f)

        torch.save(self.module.state_dict(), "Saved_paras/" + file_name + ".pth")
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

    def train(self, mode=0, batch_size=1, sample_training_length=1, feedback_timer=10, stop_condition=1e-2,
              stop_condition_length=100, hard_stop=4000):
        # global best_preconditioner

        counter = len(self.epoch_list)
        with torch.no_grad():
            if mode == 0:
                target = torch.rand(batch_size, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
                current_B = d_wc(target)
            elif mode == 1:
                current_B = torch.rand(NUM_OF_LATTICES, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
                target = gmres(d_wc, current_B, best_preconditioner, best_preconditioner(current_B))[0]
            else:
                raise ValueError('Mode must be 0 or 1.')

        # Start_values
        lowest_mean = 1e6
        stop_counter = 0
        eps = stop_condition
        N = stop_condition_length
        start_time = time.time()

        try:
            self.plot_data()
            while True:
                # Creating new sample each n-th round
                if counter % sample_training_length == 0:
                    with torch.no_grad():
                        if mode == 0:
                            target = torch.rand(batch_size, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF,
                                                dtype=torch.complex64)
                            current_B = d_wc(target)
                        elif mode == 1:
                            current_B = torch.rand(NUM_OF_LATTICES, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF,
                                                   dtype=torch.complex64)
                            target = gmres(d_wc, current_B, best_preconditioner, best_preconditioner(current_B))[0]

                # zeroing gradients
                self.optimizer.zero_grad()

                # Calculating the loss
                loss = loss_fn(self.module(current_B), target)

                # Backward pass
                loss.backward()

                # changing parameters
                self.optimizer.step()

                counter += 1

                # Appending epoch and loss values to lists
                self.epoch_list.append(counter)
                self.loss_list.append(loss.item())
                if (counter - 1) % feedback_timer == 0:
                    end_time = time.time()
                    clear_output(wait=True)
                    execution_time = end_time - start_time
                    # print(f"Time since last Update: {execution_time * 1e3:.3f} ms")
                    self.plot_data()
                    start_time = time.time()

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
                time.sleep(0.1)

            clear_output(wait=True)
            self.plot_data()
            print("\n" + "Goal reached!")

        except KeyboardInterrupt:
            clear_output(wait=True)
            self.plot_data()
            print("\n" + "Training stopped manually!")
