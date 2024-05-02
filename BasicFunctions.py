from Constants import *


def dagger(matrix):
    conj_matrix = torch.conj(matrix)
    out = conj_matrix.transpose(-2, -1)
    return out


def stretch(field):
    lattice_num_points = 1
    for number in LATTICE:
        lattice_num_points *= number
    out = field.reshape(NUM_OF_LATTICES, lattice_num_points, GAUGE_DOF, -1)
    return out


def contract(field):
    out = field.reshape(NUM_OF_LATTICES, *LATTICE, GAUGE_DOF, -1)
    return out


def gauge_tra(field, new_gauge, field_is_gauge_field=False):
    out_field = torch.matmul(new_gauge, field)
    if field_is_gauge_field:
        for index in range(out_field.shape[0]):
            out_field[index] = torch.matmul(out_field[index], torch.roll(dagger(new_gauge), shifts=-1, dims=index))
    return out_field


    # try diffrent pathes
def make_super_gauge_field(gauge_field):
    lattice = gauge_field.shape[1:-2]
    super_gauge_field = torch.zeros(*lattice, *gauge_field.shape[1:], dtype=torch.complex128)
    super_gauge_field[0, 0, 0, 0] = torch.diag_embed(torch.ones(*lattice, gauge_field.shape[-1]))

    for x_dim in range(1, lattice[0]):
        gauge = dagger(torch.roll(gauge_field[0], shifts=-(x_dim - 1), dims=0))
        super_gauge_field[x_dim, 0, 0, 0] = torch.matmul(gauge, super_gauge_field[x_dim - 1, 0, 0, 0])
    for y_dim in range(1, lattice[1]):
        gauge = dagger(torch.roll(gauge_field[1], shifts=-(y_dim - 1), dims=1))
        super_gauge_field[0, y_dim, 0, 0] = torch.matmul(gauge, super_gauge_field[0, y_dim - 1, 0, 0])
    for z_dim in range(1, lattice[2]):
        gauge = dagger(torch.roll(gauge_field[2], shifts=-(z_dim - 1), dims=2))
        super_gauge_field[0, 0, z_dim, 0] = torch.matmul(gauge, super_gauge_field[0, 0, z_dim - 1, 0])
    for t_dim in range(1, lattice[3]):
        gauge = dagger(torch.roll(gauge_field[3], shifts=-(t_dim - 1), dims=3))
        super_gauge_field[0, 0, 0, t_dim] = torch.matmul(gauge, super_gauge_field[0, 0, 0, t_dim - 1])
        

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
    
    super_gauge_field = torch.einsum("xyztayztil,ayztabztlk,abztabctks,abctabcdsj->xyztabcdij", [super_gauge_field, super_gauge_field, super_gauge_field, super_gauge_field])

    return super_gauge_field
