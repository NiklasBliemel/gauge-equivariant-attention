from TransformerModules1 import Transformer1, PTC
from TransformerModules import Transformer
import pickle
import torch


def transformer1(file):
    with open("Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = Transformer1(*structure)
    out.load_state_dict(torch.load("Saved_paras/" + file + ".pth"))
    return out

def transformer(file):
    with open("Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = Transformer(*structure)
    out.load_state_dict(torch.load("Saved_paras/" + file + ".pth"))
    return out

def ptc_transformer(file):
    with open("Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = PtcTransformer(*structure)
    out.load_state_dict(torch.load("Saved_paras/" + file + ".pth"))
    return out

def ptc(file):
    with open("Saved_structures/" + file + ".pkl", 'rb') as f:
        structure = pickle.load(f)
    out = PTC(*structure)
    out.load_state_dict(torch.load("Saved_paras/" + file + ".pth"))
    return out
