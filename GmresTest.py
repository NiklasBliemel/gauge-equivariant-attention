from configuration.GMRES import gmres
from configuration.Operators import D_WC, nn
from configuration.LoadData import load_trained_module
import matplotlib.pyplot as plt


def gmres_test(Module: nn.Module = None, saved_module_name=None, small=False, max_iter=1000, tol=1e-2):
    b = create_sample(small)
    operator = init_dwc(small)

    if Module is None or saved_module_name is None:
        preconditioner = load_trained_module(Module, saved_module_name)
        for param in preconditioner.parameters():  # Disable grad calculation for better performance
            param.requires_grad = False
        field, steps, res_list, time_taken = gmres(operator, b, preconditioner, preconditioner(b), max_iter=max_iter,
                                                   tol=tol, print_res=False)

    else:
        field, steps, res_list, time_taken = gmres(operator, b, max_iter=max_iter, tol=tol, print_res=False)

    error = torch.norm((operator(field) - b).view(-1))
    safe_result_as_png(res_list, steps, error, time_taken)


def safe_result_as_png(res_list, steps, error, time_taken):
    step_list = list(range(1, steps + 1))
    plt.plot(step_list, res_list, marker='o', linestyle='-', markersize=0.1)
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.ylabel('Residual')
    plt.title(f"\nUsing GMRES: error: {error:.3e} in time {time_taken:.1f}ms and {steps} iterations")
    plt.grid(True)
    plt.savefig("plot.png")


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
