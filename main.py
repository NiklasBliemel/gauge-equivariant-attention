from config.Constants import *
from config.TransformerModules import Transformer, PTC, SuperPtc
from Training import train_module
from GmresTest import gmres_test
from time import time

"""""
This is the main script for running code. It is designed to be used on the command line. 
Functions that are designed for main:
* train_module
* gmres_test
"""""

torch.cuda.empty_cache()
print("Starting PTC training!")
time_point = time()
train_module(PTC, structure=(NON_GAUGE_DOF, NON_GAUGE_DOF, PTC_PATHES, GAUGE_FIELD_SMALL), small=True)
print(f"Finished PTC training in {(time() - time_point):.1f} s")

torch.cuda.empty_cache()
print("Starting SuperPtc training!")
time_point = time()
train_module(SuperPtc, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF), small=True)
print(f"Finished SuperPtc training in {(time() - time_point):.1f} s")

torch.cuda.empty_cache()
print("Starting Transformertraining!")
time_point = time()
train_module(Transformer, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF, 16), small=True)
print(f"Finished Transformer training in {(time() - time_point):.1f} s")
