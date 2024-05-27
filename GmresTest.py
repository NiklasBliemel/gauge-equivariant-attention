from config.Constants import *
from config.GMRES import gmres, gmres_precon
from config.Operators import D_WC, nn
from config.LoadData import load_trained_module
import matplotlib.pyplot as plt


def gmres_test(Module: nn.Module = None, saved_module_name=None, small=False, max_iter=1000, tol=1e-3):
    b = create_sample(small)
    operator = init_dwc(small)

    if Module is not None and saved_module_name is not None:
        preconditioner = load_trained_module(Module, saved_module_name)
        for param in preconditioner.parameters():  # Disable grad calculation for better performance
            param.requires_grad = False
        field, steps, time_taken = gmres_precon(operator, b, preconditioner, max_iter=max_iter, tol=tol, print_res=True)

    else:
        saved_module_name = "none"
        field, steps, time_taken = gmres(operator, b, max_iter=max_iter, tol=tol, print_res=True)

    error = torch.norm((operator(field) - b).view(-1))
    print(f"module: {saved_module_name} | error: {error:.3e} | time: {time_taken:.0f}ms | steps: {steps} iterations")


def init_dwc(small):
    if small:
        operator = D_WC(M, GAUGE_FIELD_SMALL)
    else:
        operator = D_WC(M, GAUGE_FIELD)
    return operator


def create_sample(small):
    if small:
        b = torch.rand(DEFAULT_BATCH_SIZE, *LATTICE_SMALL, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
    else:
        b = torch.rand(DEFAULT_BATCH_SIZE, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
    return b
