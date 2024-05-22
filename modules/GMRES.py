from modules.BasicFunctions import *
from time import time
import matplotlib.pyplot as plt

"""""
Generalized Method of Residuals is a iterative method to solve a system of linear equations defined as Dx = b, where
D is a diagonal heavy Matrix and x, b are vectors. In Lattice QCD the Dirac Wilson Clover operator Dwc can be regarded
as a square Matrix for a flattened L-QCD color field. The code implements such a Solver for torch tensors, specified for
solving Dwc(x)=b. 
"""""


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
