from modules.BasicFunctions import *
from time import time
import matplotlib.pyplot as plt


def gmres(operator, b, preconditioner=None, x0=None, k=100, tol=1e-3, print_res=False):
    res_list = []
    n = b.numel()
    b_shape = b.shape
    H = torch.zeros(k + 1, k, dtype=torch.complex64)
    V = torch.zeros(n, k + 1, dtype=torch.complex64)
    Z = torch.zeros(n, k, dtype=torch.complex64)
    i_index = None
    y_i = None
    res = None

    if x0 is None:
        x0 = torch.zeros_like(b)
    r_0 = b - operator(x0)

    r_0 = r_0.view(-1)
    norm_r_0 = torch.norm(r_0)
    V[:, 0] = r_0 / norm_r_0

    for i_index in range(k):
        if preconditioner is None:
            w_temp = operator(V[:, i_index].reshape(b_shape)).view(-1)
        else:
            x_in = V[:, i_index].reshape(b_shape)
            Z[:, i_index] = preconditioner(x_in).view(-1)
            w_temp = operator(Z[:, i_index].reshape(b_shape)).view(-1)

        # ------------------------------------------------------------------------
        for j_index in range(i_index + 1):
            H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
            w_temp -= H[j_index, i_index] * V[:, j_index]
        # ------------------------------------------------------------------------

        H[i_index + 1, i_index] = torch.norm(w_temp)
        V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]

        e1 = torch.zeros(i_index + 2, dtype=torch.complex64)
        e1[0] = norm_r_0

        y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1).solution
        res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1)
        if print_res:
            print(f"res-{i_index+1}: {res:.3e}")
        res_list.append(res.item())
        if res < tol:
            if preconditioner is None:
                x_out = x0 + torch.matmul(V[:, :i_index + 1], y_i).reshape(b_shape)
            else:
                x_out = x0 + torch.matmul(Z[:, :i_index + 1], y_i).reshape(b_shape)

            return x_out, res, i_index + 1, res_list

    if preconditioner is None:
        x_out = x0 + torch.matmul(V[:, :i_index + 1], y_i).reshape(b_shape)
    else:
        x_out = x0 + torch.matmul(Z[:, :i_index + 1], y_i).reshape(b_shape)

    return x_out, res, i_index + 1, res_list


class GmresTest:
    def __init__(self, operator, num_of_lattices, lattice, gauge_dof, non_gauge_dof, max_iter=100, tol=1e-2):
        self.operator = operator
        self.max_iter = max_iter
        self.tol = tol
        self.num_of_lattices = num_of_lattices
        self.lattice = lattice
        self.gauge_dof = gauge_dof
        self.non_gauge_dof = non_gauge_dof

    def __call__(self, preconditioner=None, print_res=True):
        b = torch.rand(self.num_of_lattices, *self.lattice, self.gauge_dof, self.non_gauge_dof,
                       dtype=torch.complex64)
        x0 = None
        if preconditioner is not None:
            x0 = preconditioner(b)
        time_before = time()
        field, res, steps, res_list = gmres(self.operator, b, preconditioner, x0, self.max_iter, self.tol, print_res)
        time_taken = (time() - time_before) * 1e3
        print(f"res: {res:.3e}")
        error = torch.norm((self.operator(field) - b).view(-1))
        print(f"Using GMRES: error: {error:.3e} in time {time_taken:.1f}ms and {steps} iterations")

        steps_list = list(range(1, steps + 1))
        plt.plot(steps_list, res_list, marker='o', linestyle='-', markersize=0.1)
        plt.xlabel('Iteration')
        plt.yscale('log')
        plt.ylabel('Residual')
        plt.title("Residual - Plot")
        plt.grid(True)
        plt.show()
