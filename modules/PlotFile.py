import matplotlib.pyplot as plt
import torch


def plot_file(name, logarithmic=True):
    plot_data = "modules/Saved_plots/" + name + ".txt"
    
    epoch_list = []
    loss_list = []
    
    with open(plot_data, 'r') as file:
        for line in file:
            epoch, loss = line.strip().split('\t')
            epoch_list.append(int(epoch))
            loss_list.append(float(loss))
            
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
        plt.show()
        