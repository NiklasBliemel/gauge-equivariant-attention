import torch

from config.BasicFunctions import *
from config.LqcdOperators import nn
from time import time

"""""
Generalized Method of Residuals is a iterative method to solve a system of linear equations defined as Dx = b, where
D is a diagonal heavy Matrix and x, b are vectors. In Lattice QCD the Dirac Wilson Clover operator Dwc can be regarded
as a square Matrix for a flattened L-QCD field. The code implements such a Solver for torch tensors, specified for
solving Dwc(x)=b. 
"""""


def gmres(operator: nn.Module, b: torch.Tensor, max_iter=100, tol=1e-2):
    time_before = time()
    y_i = torch.zeros(0)
    i_index = 0

    b_shape = b.shape
    H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex64)
    V = torch.zeros(b.numel(), max_iter + 1, dtype=torch.complex64)
    e1 = torch.zeros(max_iter + 1, dtype=torch.complex64)
    with torch.no_grad():
        r0 = b.view(-1)
        e1[0] = torch.norm(r0)
        V[:, 0] = b.view(-1) / e1[0]

        for i_index in range(max_iter):
            w_temp = operator(V[:, i_index].reshape(b_shape)).view(-1)

            for j_index in range(i_index + 1):
                H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
                w_temp -= H[j_index, i_index] * V[:, j_index]
            H[i_index + 1, i_index] = torch.norm(w_temp)
            V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]

            y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1[:i_index + 2]).solution
            res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1[:i_index + 2])

            if res < tol:
                break

        x_out = torch.matmul(V[:, :(i_index + 1)], y_i).reshape(b_shape)
    return x_out, i_index + 1, (time() - time_before) * 1e3


def gmres_precon(operator: nn.Module, b: torch.Tensor, preconditioner: nn.Module, max_iter=100, tol=1e-2):
    time_before = time()
    y_i = torch.zeros(0)
    i_index = 0

    n = b.numel()
    b_shape = b.shape
    H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex64)
    V = torch.zeros(n, max_iter + 1, dtype=torch.complex64)
    e1 = torch.zeros(max_iter + 1, dtype=torch.complex64)
    with torch.no_grad():
        x0 = preconditioner(b)
        r_0 = (b - operator(x0)).view(-1)
        e1[0] = torch.norm(r_0)
        V[:, 0] = r_0 / e1[0]

        for i_index in range(max_iter):
            w_temp = preconditioner(V[:, i_index].reshape(b_shape)).view(-1)
            w_temp = operator(w_temp.reshape(b_shape)).view(-1)

            for j_index in range(i_index + 1):
                H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
                w_temp -= H[j_index, i_index] * V[:, j_index]
            H[i_index + 1, i_index] = torch.norm(w_temp)
            V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]

            y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1[:i_index + 2]).solution
            res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1[:i_index + 2])

            if res < tol:
                break

        x_out = x0 + torch.matmul(V[:, :(i_index + 1)], y_i).reshape(b_shape)
    return x_out, i_index + 1, (time() - time_before) * 1e3


# tol = 2e-4 because gmres diverges due to float precision
def gmres_train(operator: nn.Module, b: torch.Tensor, max_iter=100, tol=1e-3):
    i_index = 0

    b_shape = b.shape
    H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex64)
    V = torch.zeros(b.numel(), max_iter + 1, dtype=torch.complex64)
    e1 = torch.zeros(max_iter + 1, dtype=torch.complex64)
    with torch.no_grad():
        r0 = b.view(-1)
        e1[0] = torch.norm(r0)
        V[:, 0] = b.view(-1) / e1[0]

        for i_index in range(max_iter):
            w_temp = operator(V[:, i_index].reshape(b_shape)).view(-1)

            for j_index in range(i_index + 1):
                H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
                w_temp -= H[j_index, i_index] * V[:, j_index]
            H[i_index + 1, i_index] = torch.norm(w_temp)
            V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]

            y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1[:i_index + 2]).solution
            res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1[:i_index + 2])
            if res < tol:
                break
    return i_index + 1


def gmres_precon_train(operator: nn.Module, b: torch.Tensor, preconditioner: nn.Module, pure_gmres_iter: int, tol=1e-3):
    i_index = 0

    n = b.numel()
    b_shape = b.shape
    H = torch.zeros(pure_gmres_iter + 1, pure_gmres_iter, dtype=torch.complex64)
    V = torch.zeros(n, pure_gmres_iter + 1, dtype=torch.complex64)
    e1 = torch.zeros(pure_gmres_iter + 1, dtype=torch.complex64)
    with torch.no_grad():
        x0 = preconditioner(b)
        r_0 = (b - operator(x0)).view(-1)
        e1[0] = torch.norm(r_0)
        V[:, 0] = r_0 / e1[0]

        for i_index in range(pure_gmres_iter):
            w_temp = preconditioner(V[:, i_index].reshape(b_shape)).view(-1)
            w_temp = operator(w_temp.reshape(b_shape)).view(-1)

            for j_index in range(i_index + 1):
                H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
                w_temp -= H[j_index, i_index] * V[:, j_index]
            H[i_index + 1, i_index] = torch.norm(w_temp)
            V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]

            y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1[:i_index + 2]).solution
            res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1[:i_index + 2])
            if res < tol:
                break
    return (i_index + 1) / pure_gmres_iter
