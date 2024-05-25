from config.Constants import *
from config.TransformerModules import nn, Transformer, PTC, SuperPtc
from config.LoadData import load_trained_module, load_structure
from config.TrainingFunctions import DwcTrainer

"""""
Train Modules until sufficiently converged.

Structures:
* Ptc(NON_GAUGE_DOF, NON_GAUGE_DOF, PTC_PATHES, GAUGE_FIELD)
* Transformer(GAUGE_FIELD, NON_GAUGE_DOF, linear_size)
* SuperPtc(GAUGE_FIELD, NON_GAUGE_DOF)
"""""


def train_module(Module: nn.Module, structure: tuple = None, saved_module_name: str = None,
                 small: bool = False, train_with_gmres: bool = False):
    if structure is not None and saved_module_name is None:   # Make new Module
        module = Module(*structure)
        saved_module_name = choose_save_name(module, structure, small, train_with_gmres)
    elif structure is None and saved_module_name is not None:  # Load existing Module
        module = load_trained_module(Module, saved_module_name)
        structure = load_structure(saved_module_name)
    else:
        raise ValueError("You can only specify a structure or the name of a saved module")
    dwc_trainer = DwcTrainer(module, structure)
    # specify hard config of training
    dwc_trainer.train(small=small, train_with_gmres=train_with_gmres, update_plot=False)
    dwc_trainer.safe_data_as(saved_module_name)


def choose_save_name(module, structure, small, train_with_gmres):
    if isinstance(module, Transformer):
        save_name = "tr"
    elif isinstance(module, PTC):
        save_name = "ptc"
    elif isinstance(module, SuperPtc):
        save_name = "super_ptc"
    else:
        raise ValueError(f"Module {module} is not a configured Module for this function")
    for i_index in structure:
        if isinstance(i_index, int):
            save_name += "_" + str(i_index)
    if not small:
        save_name = save_name.capitalize()
    if train_with_gmres:
        save_name += "gmres"
    return save_name
