from config.Constants import *
from config.GMRES import gmres, gmres_precon
from config.LqcdOperators import D_WC, nn
from config.LoadData import load_trained_module
import matplotlib.pyplot as plt

"""""
Test a trained models performance as preconditioner in GMRES. Watch the residual evolution, compare it with pure gmres
or other models.
gmres_test() runs without preconditioner if none is given.
"""""


def gmres_test(Module: nn.Module = None, saved_module_name=None, small=False, max_iter=1000, tol=1e-3):
    b = create_sample(small)
    operator = init_dwc(small)

    if Module is not None and saved_module_name is not None:
        preconditioner = load_trained_module(Module, saved_module_name)
        field, steps, time_taken = gmres_precon(operator, b, preconditioner, max_iter=max_iter, tol=tol)

    else:
        saved_module_name = "none"
        field, steps, time_taken = gmres(operator, b, max_iter=max_iter, tol=tol)

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
