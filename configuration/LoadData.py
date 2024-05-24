from configuration.TransformerModules import nn
import pickle
import torch


def load_trained_module(Module: nn.Module, saved_module_name):
    structure = load_structure(saved_module_name)
    out = Module(*structure)
    para_data_path = "configuration/Saved_paras/" + saved_module_name + ".pth"
    out.load_state_dict(torch.load(para_data_path))
    return out


def load_plot_data(saved_module_name):
    epoch_list = []
    loss_list = []
    plot_data_path = "configuration/Saved_plots/" + saved_module_name + ".txt"
    with open(plot_data_path, 'r') as file:
        for line in file:
            epoch, loss = line.strip().split('\t')
            epoch_list.append(int(epoch))
            loss_list.append(float(loss))
    return epoch_list, loss_list


def load_structure(saved_module_name):
    structure_data_path = "configuration/Saved_structures/" + saved_module_name + ".pkl"
    with open(structure_data_path, 'rb') as f:
        structure = pickle.load(f)
    return structure
