import torch.nn as nn
from BasicFunctions import *


# Hop and Path operator

class H(nn.Module):
    def __init__(self, dimension, gauge_field):
        super(H, self).__init__()
        self.direction = dimension
        self.gauge_field = gauge_field
        self.lattice_dim = len(gauge_field.shape[1:-2])

    def forward(self, field):
        if self.direction == 0:
            return field
        out_field = torch.roll(field, shifts=int(self.direction / abs(self.direction)),
                               dims=(abs(self.direction) - (3 + self.lattice_dim)))
        gauge_projection = self.gauge_field[abs(self.direction) - 1]
        if self.direction > 0:
            gauge = dagger(torch.roll(gauge_projection, shifts=int(self.direction / abs(self.direction)),
                                      dims=abs(self.direction) - 1))
        else:
            gauge = gauge_projection.clone()
        out_field = torch.matmul(gauge, out_field)
        return out_field

    def gauge_tra(self, new_gauge):
        self.gauge_field = gauge_tra(self.gauge_field, new_gauge, field_is_gauge_field=True)


class T(nn.Module):
    def __init__(self, path, gauge_field):
        super(T, self).__init__()
        self.T_layer = nn.Sequential(*(H(mu, gauge_field) for mu in path))

    def forward(self, field):
        out = self.T_layer(field)
        return out

    def gauge_tra(self, new_gauge):
        for layer in self.T_layer:
            layer.gauge_tra(new_gauge)


# Specific Spin-Matrix operation

class SpinMatrix(nn.Module):
    def __init__(self, tensor):
        super(SpinMatrix, self).__init__()
        self.tensor = tensor.to(torch.complex128)

    def forward(self, field):
        out = torch.matmul(field, self.tensor)
        return out


# Dirac Wilson and Wilson-clover operator

class D_W(nn.Module):
    def __init__(self, mass, gauge_field):
        super(D_W, self).__init__()
        path_list = [0, 1, 2, 3, 4, -1, -2, -3, -4]
        tensor_list = [(mass + 4 * I_4), -(1 / 2) * (GAMMA[0] + I_4), -(1 / 2) * (GAMMA[1] + I_4),
                       -(1 / 2) * (GAMMA[2] + I_4), -(1 / 2) * (GAMMA[3] + I_4),
                       (1 / 2) * (GAMMA[0] - I_4), (1 / 2) * (GAMMA[1] - I_4), (1 / 2) * (GAMMA[2] - I_4),
                       (1 / 2) * (GAMMA[3] - I_4)]
        self.D_W_layers = nn.ModuleList([nn.Sequential(SpinMatrix(tensor_list[index]),
                                                       H(path_list[index], gauge_field)) for index in range(9)])

    def forward(self, field):
        out = torch.zeros_like(field)
        for layer in self.D_W_layers:
            out += layer(field)
        return out

    def gauge_tra(self, new_gauge):
        for layer in self.D_W_layers:
            layer[1].gauge_tra(new_gauge)


class Q(nn.Module):
    def __init__(self, mu, nu, gauge_field):
        super(Q, self).__init__()
        sum_1 = nn.Sequential(H(-mu, gauge_field), H(-nu, gauge_field),
                              H(mu, gauge_field), H(nu, gauge_field))
        sum_2 = nn.Sequential(H(-nu, gauge_field), H(mu, gauge_field),
                              H(nu, gauge_field), H(-mu, gauge_field))
        sum_3 = nn.Sequential(H(nu, gauge_field), H(-mu, gauge_field),
                              H(-nu, gauge_field), H(mu, gauge_field))
        sum_4 = nn.Sequential(H(mu, gauge_field), H(nu, gauge_field),
                              H(-mu, gauge_field), H(-nu, gauge_field))
        self.Q_layers = nn.ModuleList([sum_1, sum_2, sum_3, sum_4])

    def forward(self, field):
        out = torch.zeros_like(field)
        for layer in self.Q_layers:
            out += layer(field)
        return out

    def gauge_tra(self, new_gauge):
        for sequence in self.Q_layers:
            for layer in sequence:
                layer.gauge_tra(new_gauge)


class F(nn.Module):
    def __init__(self, mu, nu, gauge_field):
        super(F, self).__init__()
        self.q_munu = Q(mu, nu, gauge_field)
        self.q_numu = Q(nu, mu, gauge_field)

    def forward(self, field):
        out_q_munu = self.q_munu(field)
        out_q_numu = self.q_numu(field)
        out = (1 / 8) * (out_q_munu - out_q_numu)
        return out

    def gauge_tra(self, new_gauge):
        self.q_munu.gauge_tra(new_gauge)
        self.q_numu.gauge_tra(new_gauge)


class D_WC(nn.Module):
    def __init__(self, mass, gauge_field):
        super(D_WC, self).__init__()
        self.d_w = D_W(mass, gauge_field)
        self.D_WC_layers = nn.ModuleList(
            [nn.Sequential(SpinMatrix(SIGMA[mu][nu]), F(mu + 1, nu + 1, gauge_field))
             for mu in range(4) for nu in range(4)])

    def forward(self, field):
        out = self.d_w(field)
        for layer in self.D_WC_layers:
            out -= (C_SW / 4) * layer(field)
        return out

    def gauge_tra(self, new_gauge):
        self.d_w.gauge_tra(new_gauge)
        for layer in self.D_WC_layers:
            layer[1].gauge_tra(new_gauge)


d_wc = D_WC(M, GAUGE_FIELD_SMALL)
