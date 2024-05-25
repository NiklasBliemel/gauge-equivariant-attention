from config.Constants import *
from config.TransformerModules import Transformer, PTC, SuperPtc
from Training import train_model
from GmresTest import gmres_test

train_model(PTC, structure=(NON_GAUGE_DOF, NON_GAUGE_DOF), small=True, train_with_gmres=False)

train_model(SuperPtc, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF), small=True, train_with_gmres=False)

train_model(Transformer, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF, 16), small=True, train_with_gmres=True)

train_model(PTC, structure=(NON_GAUGE_DOF, NON_GAUGE_DOF), small=True, train_with_gmres=True)

train_model(SuperPtc, structure=(GAUGE_FIELD_SMALL, NON_GAUGE_DOF), small=True, train_with_gmres=True)
