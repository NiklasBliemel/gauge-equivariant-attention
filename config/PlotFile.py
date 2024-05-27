import matplotlib.pyplot as plt
import torch
from config.LoadData import load_plot_data, load_gmres_data


def plot_file(file_name):
    epoch_list, loss_list = load_plot_data(file_name)
    plot(file_name, epoch_list, loss_list)


def plot(name, epoch_list, loss_list):
    plt.plot(epoch_list, loss_list, marker='o', linestyle='-', markersize=0.1)
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
    plt.show()


def sigma_plot():
    epoch_list_ptc, loss_list_ptc = load_plot_data('ptc_4_4')
    epoch_list_super_ptc, loss_list_super_ptc = load_plot_data('super_ptc_4')
    epoch_list_tra, loss_list_tra = load_plot_data('tr_4_16')

    gmres_epochs_ptc, gmres_itergain_ptc = load_gmres_data('ptc_4_4')
    gmres_epochs_sptc, gmres_itergain_sptc = load_gmres_data('super_ptc_4')
    gmres_epochs_tra, gmres_itergain_tra = load_gmres_data('tr_4_16')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax1.plot(epoch_list_ptc, loss_list_ptc, 'b', label='PTC', linestyle='-')
    ax1.plot(epoch_list_super_ptc, loss_list_super_ptc, 'g', label='SuperPtc', linestyle='-')
    ax1.plot(epoch_list_tra, loss_list_tra, 'r', label='Attention', linestyle='-')
    ax1.set_yscale('log')
    ax1.set_ylabel('Cost function')
    ax1.legend()

    ax2.plot(gmres_epochs_ptc, gmres_itergain_ptc, 'b', label='PTC', linestyle='', marker='^', markersize=5)
    ax2.plot(gmres_epochs_sptc, gmres_itergain_sptc, 'g', label='SuperPtc', linestyle='', marker='x', markersize=5)
    ax2.plot(gmres_epochs_tra, gmres_itergain_tra, 'r', label='Attention', linestyle='', marker='s', markersize=5)
    ax2.set_xlabel('Training step')
    ax2.set_ylabel('GMRES Iterations')
    ax2.legend()
    plt.show()
