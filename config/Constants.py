import torch
import numpy as np


"""""
Here we define all the important global constants defining the L-QCD config.
In addition the used gauge-field and the gamma-matrices, c_sw-constant and mass-constant for Dwc are defined here.
Also the Path's used by the PTC module are here predefined.
"""""


DEFAULT_BATCH_SIZE = 1
LATTICE = (8, 8, 8, 16)
LATTICE_SMALL = (4, 4, 4, 8)

# Field dimensions
GAUGE_DOF = 3
NON_GAUGE_DOF = 4

# PTC pathes:
PTC_PATHES = [[0], [1], [-1], [2], [-2], [3], [-3], [4], [-4]]

# Wilson Dirac constants
I_4 = torch.eye(4).to(torch.complex64)
M = (-0.6 * I_4).to(torch.complex64)
data = np.load("config/npy_files/gamma_matrices.npy").tolist()
GAMMA = torch.tensor(data).to(torch.complex64)

# Gauge-Field; shape(4,8,8,8,16,3,3)
data = np.load("config/npy_files/U.npy").tolist()
GAUGE_FIELD = torch.tensor(data).to(torch.complex64)

# Gauge-Field; shape(4,4,4,4,8,3,3)
GAUGE_FIELD_SMALL = torch.empty(4, 4, 4, 4, 8, 3, 3, dtype=torch.complex64)
for i in range(4):
    direction = [1 if j == i else 0 for j in range(4)]
    for x in range(4):
        for y in range(4):
            for z in range(4):
                for t in range(8):
                    GAUGE_FIELD_SMALL[i, x, y, z, t] = torch.matmul(GAUGE_FIELD[i, 2 * x, 2 * y, 2 * z, 2 * t],
                                                                    GAUGE_FIELD[i, 2 * x + direction[0], 
                                                                                   2 * y + direction[1], 
                                                                                   2 * z + direction[2], 
                                                                                   2 * t + direction[3]])

# Wilson-clover constants
C_SW = 1.0
SIGMA = torch.empty(4, 4, 4, 4, dtype=torch.complex64)
for i in range(4):
    for j in range(4):
        SIGMA[i][j] = (1 / 2) * (torch.matmul(GAMMA[i], GAMMA[j]) - torch.matmul(GAMMA[j], GAMMA[i]))
        
one_random_field = torch.rand(1, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
