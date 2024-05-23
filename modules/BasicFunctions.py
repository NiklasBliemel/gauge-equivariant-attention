from modules.Constants import *


"""""
Here basic operations like dagger are implemented for torch tensors.
Also gauge transformation is defined for both field and gauge-field of any size.
Make super_gauge_field is a function that takes a gauge-field and returns a large volume squared gauge-field that can
connect any point from a field with another over a predefined path trough the original gauge_field
"""""


def dagger(matrix):
    conj_matrix = torch.conj(matrix)
    out = conj_matrix.transpose(-2, -1)
    return out


def gauge_tra(field, new_gauge, field_is_gauge_field=False):
    out_field = torch.matmul(new_gauge, field)
    if field_is_gauge_field:
        for index in range(out_field.shape[0]):
            out_field[index] = torch.matmul(out_field[index], torch.roll(dagger(new_gauge), shifts=-1, dims=index))
    return out_field


# shortest path's (order: t -> z -> y -> x)
def make_super_gauge_field(gauge_field):
    lattice = gauge_field.shape[1:-2]
    super_gauge_field = torch.zeros(*lattice, *gauge_field.shape[1:], dtype=torch.complex64)
    super_gauge_field[0, 0, 0, 0] = torch.diag_embed(torch.ones(*lattice, gauge_field.shape[-1]))

    for x_dim in range(1, lattice[0]):
        if x_dim <= lattice[0] // 2:
            gauge = dagger(torch.roll(gauge_field[0], shifts=1 - x_dim, dims=0))
            super_gauge_field[x_dim, 0, 0, 0] = torch.matmul(gauge, super_gauge_field[x_dim - 1, 0, 0, 0])
        else:
            gauge = torch.roll(gauge_field[0], shifts=x_dim - lattice[0] // 2, dims=0)
            super_gauge_field[(lattice[0] // 2 - x_dim) % lattice[0], 0, 0, 0] = torch.matmul(
                gauge, super_gauge_field[(lattice[0] // 2 - x_dim + 1) % lattice[0], 0, 0, 0])
    for y_dim in range(1, lattice[1]):
        if y_dim <= lattice[1] // 2:
            gauge = dagger(torch.roll(gauge_field[1], shifts=1 - y_dim, dims=1))
            super_gauge_field[0, y_dim, 0, 0] = torch.matmul(gauge, super_gauge_field[0, y_dim - 1, 0, 0])
        else:
            gauge = torch.roll(gauge_field[1], shifts=y_dim - lattice[1] // 2, dims=1)
            super_gauge_field[0, (lattice[1] // 2 - y_dim) % lattice[1], 0, 0] = torch.matmul(
                gauge, super_gauge_field[0, (lattice[1] // 2 - y_dim + 1) % lattice[1], 0, 0])
    for z_dim in range(1, lattice[2]):
        if z_dim <= lattice[2] // 2:
            gauge = dagger(torch.roll(gauge_field[2], shifts=1 - z_dim, dims=2))
            super_gauge_field[0, 0, z_dim, 0] = torch.matmul(gauge, super_gauge_field[0, 0, z_dim - 1, 0])
        else:
            gauge = torch.roll(gauge_field[2], shifts=z_dim - lattice[2] // 2, dims=2)
            super_gauge_field[0, 0, (lattice[2] // 2 - z_dim) % lattice[2], 0] = torch.matmul(
                gauge, super_gauge_field[0, 0, (lattice[2] // 2 - z_dim + 1) % lattice[2], 0])
    for t_dim in range(1, lattice[3]):
        if t_dim <= lattice[3] // 2:
            gauge = dagger(torch.roll(gauge_field[3], shifts=1 - t_dim, dims=3))
            super_gauge_field[0, 0, 0, t_dim] = torch.matmul(gauge, super_gauge_field[0, 0, 0, t_dim - 1])
        else:
            gauge = torch.roll(gauge_field[3], shifts=t_dim - lattice[3] // 2, dims=3)
            super_gauge_field[0, 0, 0, (lattice[3] // 2 - t_dim) % lattice[3]] = torch.matmul(
                gauge, super_gauge_field[0, 0, 0, (lattice[3] // 2 - t_dim + 1) % lattice[3]])

    dimensions = [torch.arange(lattice[0]), torch.arange(lattice[1]), torch.arange(lattice[2]),
                  torch.arange(lattice[3])]

    # Create grid of indices
    x_shift, y_shift, z_shift, t_shift, x_orig, y_orig, z_orig, t_orig = torch.meshgrid(*dimensions, *dimensions,
                                                                                        indexing="ij")

    # Compute indices for t1
    x_goal, y_goal, z_goal, t_goal = (x_shift - x_orig) % lattice[0], (y_shift - y_orig) % lattice[1], (
            z_shift - z_orig) % lattice[2], (t_shift - t_orig) % lattice[3]

    # Index t1 with computed indices
    super_gauge_field = super_gauge_field[x_goal, y_goal, z_goal, t_goal, x_orig, y_orig, z_orig, t_orig]

    super_gauge_field = torch.einsum("xyztayztil,ayztabztlk,abztabctks,abctabcdsj->xyztabcdij",
                                     [super_gauge_field, super_gauge_field, super_gauge_field, super_gauge_field])

    return super_gauge_field
