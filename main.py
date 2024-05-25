from config.Constants import *
from config.TransformerModules import Transformer, PTC, SuperPtc
from Training import train_module
from GmresTest import gmres_test

"""""
This is the main script for running code. It is designed to be used on the command line. 
Functions that are designed for main (save data when finished):
* train_module
* gmres_test
"""""


train_module(PTC, structure=(NON_GAUGE_DOF, NON_GAUGE_DOF), small=True, train_with_gmres=False)

train_module(SuperPtc, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF), small=True, train_with_gmres=False)

train_module(Transformer, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF, 16), small=True, train_with_gmres=True)

train_module(PTC, structure=(NON_GAUGE_DOF, NON_GAUGE_DOF), small=True, train_with_gmres=True)

train_module(SuperPtc, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF), small=True, train_with_gmres=True)
