from modules.TransformerModules import Transformer, NonDirectionalTransformer, PTC
import pickle
import torch


def transformer(file):
    with open("modules/Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = Transformer(*structure)
    out.load_state_dict(torch.load("modules/Saved_paras/" + file + ".pth"))
    return out


def non_direc_transformer(file):
    with open("modules/Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = NonDirectionalTransformer(*structure)
    out.load_state_dict(torch.load("modules/Saved_paras/" + file + ".pth"))
    return out


def ptc(file):
    with open("modules/Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = PTC(*structure)
    out.load_state_dict(torch.load("modules/Saved_paras/" + file + ".pth"))
    return out
