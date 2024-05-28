from nnModules import nn, GaugeCovAttention, PTC, SuperPtc
from config.TrainingFunctions import DwcTrainer

"""""
Train Modules until sufficiently converged. Test iteration gain with gmres after each 50 training steps or
when converging.
To train a new promising model, import it from nnModules.py and add it to choose_save_name

Structures:
* Ptc(NON_GAUGE_DOF, NON_GAUGE_DOF, PTC_PATHES, GAUGE_FIELD)
* Transformer(GAUGE_FIELD, NON_GAUGE_DOF, linear_size)
* SuperPtc(GAUGE_FIELD, NON_GAUGE_DOF)
"""""


def train_module(Module: nn.Module, structure: tuple, small=False):
    module = Module(*structure)
    save_name = choose_save_name(module, structure, small)
    dwc_trainer = DwcTrainer(module, structure)
    dwc_trainer.scripted_training(small=small)
    dwc_trainer.save_data_as(save_name)
    dwc_trainer.save_itergain_plot(save_name + "_iter")


def choose_save_name(module, structure, small):
    if isinstance(module, GaugeCovAttention):
        save_name = "at"
    elif isinstance(module, PTC):
        save_name = "ptc"
    elif isinstance(module, SuperPtc):
        save_name = "super_ptc"
    else:
        raise ValueError(f"Module {module} is not a configured Module for this function")
    for num in structure:
        if isinstance(num, int):
            save_name += "_" + str(num)
    if not small:
        save_name += "_big"
    return save_name
