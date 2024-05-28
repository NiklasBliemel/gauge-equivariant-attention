from nnModules import GaugeCovAttention, PTC, SuperPtc
from config.GmresTest import gmres_test

"""""
This is the main script for running computational heavy code. Make sure to empty cache when possible to avoid out of 
memory error. [Tip: cuda.empty_cache()]

Functions that are designed for main:
* train_module(ModuleClassName, structure, small=True/False) - scripted training of module, auto-saving data afterwards
* gmres_test(ModuleClassName, "saved_file_name", small=True/False) - for testing module as preconditioner for gmres
"""""

gmres_test(small=True)
gmres_test(GaugeCovAttention, "at_4_16", small=True)
