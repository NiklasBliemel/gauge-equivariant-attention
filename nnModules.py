from config.BasicFunctions import *
from config.LqcdOperators import nn, T
import time


"""""
This is the core part of the code. Here all the Nural Network Models are implemented as subclasses of nn.Module form 
torch.
The main goal is to construct a gauge equivalent attention model, which ultimate is used to work as a preconditioner 
for solving Dwc(x)=b with Gmres.
For more detailed information on how the Attention model works, refer to:
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
In addition the the attention model, also the Parallel Transport Convolution Layer is implemented here for comparison.
Details to that are found in https://arxiv.org/pdf/2302.05419

This section shall also serve as a testing ground for improvements and new ideas for the attention model or other
architectures.
 
Quick guide to implement your own model:
* Study the pytorch nn library
* Use implemented submodules or create your own to construct your own architecture
* Use covariance_test from CovarianceTest.py to test if your architecture maintains gauge symmetries
* Train your model using DwcTrainer class in TrainingFunctions.py
* If trained sufficiently use gmres_test form GmresTest.py to test your models performance as preconditioner
* If your model works, use train_module form Training.py to compare it with the other models. Make sure to add your 
  models class to Training.py as described in that file.
* Include your model to itergain_plot from PlotFile.py and use it to generate a plot for comparison
"""""


# Random Spin-matrix, same for every site
class NonGaugeLinear(nn.Module):
    def __init__(self, input_dof, output_dof):
        super(NonGaugeLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dof, output_dof, dtype=torch.complex64))
        self.input_dof = input_dof

    def forward(self, field):
        out = torch.matmul(field, self.weights)
        return out


# Starts as dupe Matrix, because default pe is already close to goal
class ReducedNonGaugeLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReducedNonGaugeLinear, self).__init__()
        assert input_size % 4 == 0, 'input_size must be divisible by 4'

        block_width = input_size // 4
        block_height = output_size // 4

        # Create matrix-blocks (start with one)
        block = torch.zeros(block_width, block_height, dtype=torch.complex64)
        for index in range(output_size // 4):
            block[index % block_width][index] = 1

        # Create the diagonal block matrix
        block_matrix = torch.zeros(input_size, output_size, dtype=torch.complex64)
        block_matrix[:block_width, :block_height] = block
        block_matrix[block_width:2 * block_width, block_height:2 * block_height] = block
        block_matrix[2 * block_width:3 * block_width, 2 * block_height:3 * block_height] = block
        block_matrix[3 * block_width:4 * block_width, 3 * block_height:4 * block_height] = block

        self.weights = nn.Parameter(block_matrix)

    def forward(self, field):
        out = torch.matmul(field, self.weights)
        return out


# Volume size spin matrix (different at each site)
class LocalNonGaugeLinear(nn.Module):
    def __init__(self, input_dof, lattice):
        super(LocalNonGaugeLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(*lattice, input_dof, input_dof, dtype=torch.complex64))

    def forward(self, field):
        out = torch.matmul(field, self.weights)
        return out


# Volume size Matrix which transforms Values based on relative shift compared to Queries
class PathBasedLinear(nn.Module):
    def __init__(self, non_gauge_dof, lattice):
        super(PathBasedLinear, self).__init__()
        self.volume = 1
        for num in lattice:
            self.volume *= num
        local_spin_matrix = torch.randn(self.volume, non_gauge_dof, non_gauge_dof, dtype=torch.complex64)
        n, self.m = torch.meshgrid(torch.arange(self.volume), torch.arange(self.volume), indexing="ij")
        self.n_new = (n - self.m) % self.volume
        self.weights = nn.Parameter(local_spin_matrix)

    def forward(self, field):
        out = field.reshape(field.shape[0], self.volume, *field.shape[-2:])
        out = torch.einsum('Nmis,nsj->Nnmij', [out, self.weights])
        out = out[:, self.n_new, self.m]
        return out


class PTC(nn.Module):
    def __init__(self, input_dof, output_dof, path_list, gauge_field):
        super(PTC, self).__init__()
        self.PTC_layers = nn.ModuleList(
            [nn.Sequential(NonGaugeLinear(input_dof, output_dof), T(p, gauge_field)) for p
             in path_list])
        self.num_pathes = len(path_list)
        self.input_dof = input_dof

    def forward(self, field):
        out = self.PTC_layers[0](field)
        for index in range(1, len(self.PTC_layers)):
            out += self.PTC_layers[index](field)
        # print("PTC mean: ", torch.mean(torch.abs(out)))
        return out / (self.num_pathes * self.input_dof) ** 0.5

    def gauge_tra(self, new_gauge):
        for layer in self.PTC_layers:
            layer[1].gauge_tra(new_gauge)


# Quadruples input field and adds a four block PE-Matrix, each block representing one Dimension
class PE_4D(nn.Module):
    def __init__(self, gauge_field, input_non_gauge_dof):
        super(PE_4D, self).__init__()
        X_dim, Y_dim, Z_dim, T_dim = gauge_field.shape[1:-2]
        gauge_dof = gauge_field.shape[-1]

        x_indices = torch.arange(X_dim, dtype=torch.float32)
        t_indices = torch.arange(T_dim, dtype=torch.float32)
        k_indices = torch.arange(2, input_non_gauge_dof + 2, dtype=torch.float32)

        # Compute values using vectorized operations
        x_block_even = torch.sin(2 * x_indices.unsqueeze(-1) / X_dim ** (k_indices[::2] / input_non_gauge_dof))
        t_block_even = torch.sin(2 * t_indices.unsqueeze(-1) / T_dim ** (k_indices[::2] / input_non_gauge_dof))
        x_block_odd = torch.cos(2 * x_indices.unsqueeze(-1) / X_dim ** ((k_indices[1::2] - 1) / input_non_gauge_dof))
        t_block_odd = torch.cos(2 * t_indices.unsqueeze(-1) / T_dim ** ((k_indices[1::2] - 1) / input_non_gauge_dof))

        # Combine even and odd values
        x_block = torch.zeros(X_dim, gauge_dof, input_non_gauge_dof, dtype=torch.complex64)
        t_block = torch.zeros(T_dim, gauge_dof, input_non_gauge_dof, dtype=torch.complex64)
        x_block[:, :, ::2] = x_block_even.unsqueeze(1)
        t_block[:, :, ::2] = t_block_even.unsqueeze(1)
        x_block[:, :, 1::2] = x_block_odd.unsqueeze(1)
        t_block[:, :, 1::2] = t_block_odd.unsqueeze(1)

        index_x, index_y, index_z, index_t = torch.meshgrid(torch.arange(X_dim),
                                                            torch.arange(Y_dim),
                                                            torch.arange(Z_dim),
                                                            torch.arange(T_dim), indexing="ij")

        self.pe_tensor = torch.cat((x_block[index_x],
                                    x_block[index_y],
                                    x_block[index_z],
                                    t_block[index_t]), dim=-1)

        self.dupe_tensor = torch.zeros(input_non_gauge_dof, 4 * input_non_gauge_dof, dtype=torch.complex64)
        for index_j in range(4 * input_non_gauge_dof):
            self.dupe_tensor[index_j % input_non_gauge_dof][index_j] = 1

        self.dominance = nn.Parameter(torch.tensor(1, dtype=torch.complex64), requires_grad=True)

    def forward(self, field):
        out = torch.matmul(field, self.dupe_tensor)
        out = out + self.pe_tensor.unsqueeze(0) * self.dominance
        return out

    def gauge_tra(self, new_gauge):
        self.pe_tensor = gauge_tra(self.pe_tensor, new_gauge)


# Direction based linear + Super-gauge-field
class SuperPtc(nn.Module):
    def __init__(self, gauge_field, input_non_gauge_dof):
        super(SuperPtc, self).__init__()
        self.input_non_gauge_dof = input_non_gauge_dof
        lattice = gauge_field.shape[1:-2]
        self.volume = 1
        for num in lattice:
            self.volume *= num
        self.gauge_field = gauge_field
        temp = make_super_gauge_field(gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, *gauge_field.shape[-2:])
        self.linear = PathBasedLinear(input_non_gauge_dof, lattice)

    def forward(self, field):
        out = field.reshape(field.shape[0], self.volume, *field.shape[-2:])
        out = self.linear(out)
        out = torch.einsum('nmis,Nnmsj->Nnij', [self.super_gauge_field, out]).reshape(*field.shape) / (self.volume * self.input_non_gauge_dof) ** 0.5
        # print("SuperPtc mean: ", torch.mean(torch.abs(out)))
        return out

    def gauge_tra(self, new_gauge):
        self.gauge_field = gauge_tra(self.gauge_field, new_gauge, field_is_gauge_field=True)
        temp = make_super_gauge_field(self.gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, *self.gauge_field.shape[-2:])


class SuperGaugeFieldSoftmax(nn.Module):
    def __init__(self, volume, gauge_field):
        super(SuperGaugeFieldSoftmax, self).__init__()
        self.volume = volume
        temp = make_super_gauge_field(gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, *gauge_field.shape[-2:])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, attention_scores):
        effect = self.softmax(attention_scores)
        out = torch.mul(self.super_gauge_field.permute(2, 3, 0, 1).unsqueeze(0), effect).permute(0, 3, 4, 1, 2)
        return out

    def gauge_tra(self, tra_gauge):
        temp = make_super_gauge_field(tra_gauge)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, *tra_gauge.shape[-2:])


class SelfAttention(nn.Module):
    def __init__(self, gauge_field):
        super(SelfAttention, self).__init__()
        self.gauge_field = gauge_field
        lattice = gauge_field.shape[1:-2]
        self.volume = 1
        for dim_size in lattice:
            self.volume *= dim_size
        self.activation = SuperGaugeFieldSoftmax(self.volume, self.gauge_field)

    def forward(self, queries, keys, values):
        queries_shape = queries.shape
        keys_shape = keys.shape
        # flatten Lattice dimensions of queries and keys and split spin-dim into 4 Heads
        queries = queries.reshape(queries_shape[0], self.volume, queries_shape[-2], -1, 4).permute(0, 4, 1, 2, 3)
        keys = keys.reshape(keys_shape[0], self.volume, keys_shape[-2], -1, 4).permute(0, 4, 1, 2, 3)
        # make gauge eq. h-dim^2 matrices via matmul with itself for each head
        # then flatten matrices and concat heads -> (volume x head-dim^3) Matrix (for queries and keys)
        invariant_queries = torch.real(torch.matmul(dagger(queries), queries))
        invariant_queries = invariant_queries.permute(0, 2, 3, 4, 1).reshape(queries_shape[0], self.volume, -1)
        invariant_keys = torch.real(torch.matmul(dagger(keys), keys))
        invariant_keys = invariant_keys.permute(0, 2, 3, 4, 1).reshape(keys_shape[0], self.volume, -1)
        # matmul queries and keys -> result is similar to trace(Q dagger(K) K dagger(Q)) but a lot faster
        attention_scores = torch.matmul(invariant_queries, invariant_keys.transpose(-2, -1))
        attention = self.activation((attention_scores / invariant_queries.shape[-1]) ** 0.5)
        # summ values with respective attention value and sgf path and reshape into original field shape
        out = torch.einsum("Nnmis,Nnmsj->Nnij", [attention, values])
        out = out.reshape(*queries_shape[:-1], -1)
        return out

    def gauge_tra(self, new_gauge):
        self.gauge_field = gauge_tra(self.gauge_field, new_gauge, field_is_gauge_field=True)
        self.activation.gauge_tra(self.gauge_field)


class GaugeCovAttention(nn.Module):
    def __init__(self, gauge_field, input_non_gauge_dof, linear_size):
        super(GaugeCovAttention, self).__init__()
        lattice = gauge_field.shape[1:-2]
        lattice_dof = len(lattice)
        pe_output_size = lattice_dof * input_non_gauge_dof
        self.linear_size = linear_size

        assert linear_size & lattice_dof == 0, "linear Size must be dividable by 4!"

        self.norm_field_scale = nn.Parameter(torch.tensor(7, dtype=torch.complex64))
        self.pe = PE_4D(gauge_field, input_non_gauge_dof)
        self.W_Q = ReducedNonGaugeLinear(pe_output_size, linear_size)
        self.W_K = ReducedNonGaugeLinear(pe_output_size, linear_size)
        self.W_V = PathBasedLinear(input_non_gauge_dof, lattice)
        self.self_attention = SelfAttention(gauge_field)

    def forward(self, field):
        dims = tuple(range(1, len(field.shape)))
        norm_field = self.norm_field_scale * field / torch.sum(torch.matmul(dagger(field), field), dim=dims) ** 0.5

        queries = self.W_Q(self.pe(norm_field))
        keys = self.W_K(self.pe(norm_field))
        values = self.W_V(field)

        out = self.self_attention(queries, keys, values)
        return out

    def gauge_tra(self, new_gauge):
        self.pe.gauge_tra(new_gauge)
        self.self_attention.gauge_tra(new_gauge)
