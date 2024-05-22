from modules.GMRES import gmres
from modules.Operators import D_WC
from modules.Constants import *
from modules.Preconditioner import transformer
import matplotlib.pyplot as plt
import time


def test_gmres(tra_name, small=False, max_iter=1000, tol=1e-2):
    if small:
        operator = D_WC(M, GAUGE_FIELD_SMALL)
        b = torch.rand(DEFAULT_BATCH_SIZE, *LATTICE_SMALL, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
    else:
        operator = D_WC(M, GAUGE_FIELD)
        b = torch.rand(DEFAULT_BATCH_SIZE, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)

    preconditioner = transformer(tra_name)
    time_before = time.time()
    field, res, steps, res_list = gmres(operator, b, preconditioner, preconditioner(b), max_iter, tol, True)
    time_taken = (time.time() - time_before) * 1e3
    print(f"\nres: {res:.3e}")
    error = torch.norm((operator(field) - b).view(-1))
    print(f"\nUsing GMRES: error: {error:.3e} in time {time_taken:.1f}ms and {steps} iterations")

    steps_list = list(range(1, steps + 1))
    plt.plot(steps_list, res_list, marker='o', linestyle='-', markersize=0.1)
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.ylabel('Residual')
    plt.title("Residual - Plot")
    plt.grid(True)
    plt.savefig("plot.png")


test_gmres("tr_gmres_4_16_True", small=True)
