import torch

from config.BasicFunctions import *
from config.Operators import nn
from time import time

"""""
Generalized Method of Residuals is a iterative method to solve a system of linear equations defined as Dx = b, where
D is a diagonal heavy Matrix and x, b are vectors. In Lattice QCD the Dirac Wilson Clover operator Dwc can be regarded
as a square Matrix for a flattened L-QCD field. The code implements such a Solver for torch tensors, specified for
solving Dwc(x)=b. 
"""""


def arnoldi(V, v_temp, i_index):
    for j_index in range(i_index + 1):
        h_ji = torch.vdot(v_temp, V[:, j_index])
        v_temp -= h_ji * V[:, j_index]
    h_out = torch.norm(v_temp)  # H[i +1, i]
    v_out = v_temp / h_out  # V[:, i + 1]
    return h_out, v_out


def gmres(operator: nn.Module, b: torch.Tensor, max_iter=100, tol=1e-2, print_res=False):
    time_before = time()
    y_i = torch.zeros(0)

    b_shape = b.shape
    H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex64)
    V = torch.zeros(b.numel(), max_iter + 1, dtype=torch.complex64)
    e1 = torch.zeros(max_iter + 1, dtype=torch.complex64)
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

        if print_res:
            print(f"res-{i_index + 1}: {res:.3e}")

        if res < tol:
            x_out = torch.matmul(V[:, :i_index + 1], y_i).reshape(b_shape)
            return x_out, i_index + 1, (time() - time_before) * 1e3

    x_out = torch.matmul(V[:, :max_iter], y_i).reshape(b_shape)
    return x_out, max_iter, (time() - time_before) * 1e3


def gmres_precon(operator: nn.Module, b: torch.Tensor, preconditioner: nn.Module, max_iter=100, tol=1e-2,
                 print_res=False):
    time_before = time()
    y_i = torch.zeros(0)

    n = b.numel()
    b_shape = b.shape
    H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex64)
    V = torch.zeros(n, max_iter + 1, dtype=torch.complex64)
    Z = torch.zeros(n, max_iter, dtype=torch.complex64)
    x0 = preconditioner(b)
    r_0 = (b - operator(x0)).view(-1)
    e1 = torch.zeros(max_iter + 1, dtype=torch.complex64)
    e1[0] = torch.norm(r_0)
    V[:, 0] = r_0 / e1[0]

    for i_index in range(max_iter):
        Z[:, i_index] = preconditioner(V[:, i_index].reshape(b_shape)).view(-1)
        w_temp = operator(Z[:, i_index].reshape(b_shape)).view(-1)
        
        for j_index in range(i_index + 1):
            H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
            w_temp -= H[j_index, i_index] * V[:, j_index]
        H[i_index + 1, i_index] = torch.norm(w_temp)
        V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]
        y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1[:i_index + 2]).solution
        res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1[:i_index + 2])

        if print_res:
            print(f"res-{i_index + 1}: {res:.3e}")

        if res < tol:
            x_out = x0 + torch.matmul(Z[:, :i_index + 1], y_i).reshape(b_shape)
            return x_out, i_index + 1, (time() - time_before) * 1e3

    x_out = x0 + torch.matmul(Z[:, :max_iter], y_i).reshape(b_shape)
    return x_out, max_iter, (time() - time_before) * 1e3


# tol = 1.5e-4 because it diverges in base version becasue of float percision
def gmres_train(operator: nn.Module, b: torch.Tensor, max_iter=100, tol=1e-3):

    b_shape = b.shape
    H = torch.zeros(max_iter + 1, max_iter, dtype=torch.complex64)
    V = torch.zeros(b.numel(), max_iter + 1, dtype=torch.complex64)
    e1 = torch.zeros(max_iter + 1, dtype=torch.complex64)
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
            return i_index + 1
    return max_iter


def gmres_precon_train(operator: nn.Module, b: torch.Tensor, preconditioner: nn.Module, pure_gmres_iter: int, tol=1e-3):
    
    n = b.numel()
    b_shape = b.shape
    H = torch.zeros(pure_gmres_iter + 1, pure_gmres_iter, dtype=torch.complex64)
    V = torch.zeros(n, pure_gmres_iter + 1, dtype=torch.complex64)
    Z = torch.zeros(n, pure_gmres_iter, dtype=torch.complex64)
    x0 = preconditioner(b)
    r_0 = (b - operator(x0)).view(-1)
    e1 = torch.zeros(pure_gmres_iter + 1, dtype=torch.complex64)
    e1[0] = torch.norm(r_0)
    V[:, 0] = r_0 / e1[0]

    for i_index in range(pure_gmres_iter):
        Z[:, i_index] = preconditioner(V[:, i_index].reshape(b_shape)).view(-1)
        w_temp = operator(Z[:, i_index].reshape(b_shape)).view(-1)
        
        for j_index in range(i_index + 1):
            H[j_index, i_index] = torch.vdot(w_temp, V[:, j_index])
            w_temp -= H[j_index, i_index] * V[:, j_index]
        H[i_index + 1, i_index] = torch.norm(w_temp)
        V[:, i_index + 1] = w_temp / H[i_index + 1, i_index]
        
        y_i = torch.linalg.lstsq(H[:i_index + 2, :i_index + 1], e1[:i_index + 2]).solution
        res = torch.norm(torch.matmul(H[:i_index + 2, :i_index + 1], y_i) - e1[:i_index + 2])
        if res < tol:
            return pure_gmres_iter / (i_index + 1)
    return 1.0
