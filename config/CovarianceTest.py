import torch
from config.BasicFunctions import *
from config.TransformerModules import nn, Transformer, PTC, SuperPtc

"""""
Test if your module is gauge_covariant by transforming field before and after applying module and calculating the
max absolute difference.
A module has du be initialized to run this test with.
"""""


def covariance_test(module: nn.Module):
    test_field = torch.rand(DEFAULT_BATCH_SIZE, *LATTICE_SMALL, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
    test_gauge = torch.rand(*LATTICE_SMALL, GAUGE_DOF, GAUGE_DOF, dtype=torch.complex64)
    test_gauge = torch.linalg.qr(test_gauge)[0]  # Make rand complex 3x3 Orthogonal with QR deconstruction -> unitary matrix

    output_1 = gauge_tra(module(test_field), test_gauge)  # Transform output

    module.gauge_tra(test_gauge)  # Transform module
    output_2 = module(gauge_tra(test_field, test_gauge))  # Transform input

    result = torch.max(torch.abs(output_1 - output_2))
    print(f"Test result: {result:.3e}")
