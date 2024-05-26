import matplotlib.pyplot as plt
import torch
from config.LoadData import load_plot_data


def plot_file(file_name, logarithmic=True, save_as_png=False):
    epoch_list, loss_list = load_plot_data(file_name)
    plot(file_name, epoch_list, loss_list)


def plot(name, epoch_list, loss_list):
    plt.plot(epoch_list, loss_list, marker='o', linestyle='-', markersize=0.1)
    if logarithmic:
        plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    plt.grid(True)

    last_epoch = epoch_list[-1]
    if len(loss_list) < 200:
        last_loss = loss_list[-1]
        plt.annotate(f'Last Epoch: {last_epoch}, Last Loss: {last_loss:.2f}', xy=(last_epoch, last_loss),
                     xytext=(20, 20),
                     textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))
    else:
        loss_tensor = torch.tensor(loss_list[-200:])
        loss_mean = torch.mean(loss_tensor).item()
        loss_var = torch.std(loss_tensor).item()
        plt.annotate(f'Last Epoch: {last_epoch}, Mean: {loss_mean:.2f}, Variance: {loss_var:.2f}',
                     xy=(last_epoch, loss_mean), xytext=(20, 20),
                     textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))
    if save_as_png:
        plt.savefig("Saved_plot_figures/" + name + "_plot.png")
    else:
        plt.show()
