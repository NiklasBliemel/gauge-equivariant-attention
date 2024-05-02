from Operators import *


class NonGaugeLinear(nn.Module):
    def __init__(self, input_dof, output_dof):
        super(NonGaugeLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dof, output_dof, dtype=torch.complex128))
        self.input_dof = input_dof

    def forward(self, field):
        out = torch.matmul(field, self.weights)
        return out


class ReducedNonGaugeLinear(nn.Module):
    def __init__(self, non_gauge_dof, head_dim, lattice):
        super(ReducedNonGaugeLinear, self).__init__()
        self.lattice_dof = len(lattice)
        # Create the four 4x8 blocks
        blocks = torch.zeros(non_gauge_dof, head_dim // 4, dtype=torch.complex128)
        for i in range(head_dim//4):
            blocks[i % non_gauge_dof][i] = 1
        # Create the diagonal block matrix
        block_matrix = torch.zeros(non_gauge_dof * self.lattice_dof, head_dim, dtype=torch.complex128)
        block_matrix[:non_gauge_dof, :head_dim // 4] = blocks
        block_matrix[non_gauge_dof:2 * non_gauge_dof, head_dim // 4:head_dim // 2] = blocks
        block_matrix[2 * non_gauge_dof:3 * non_gauge_dof, head_dim // 2:3 * head_dim // 4] = blocks
        block_matrix[3 * non_gauge_dof:4 * non_gauge_dof, 3 * head_dim // 4:head_dim] = blocks
        # Convert Matrix into parameters
        self.weights = nn.Parameter(block_matrix)

    def forward(self, field):
        out = torch.matmul(field, self.weights)
        return out


class LocalNonGaugeLinear(nn.Module):
    def __init__(self, input_dof, lattice_size):
        super(LocalNonGaugeLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(*lattice_size, input_dof, input_dof, dtype=torch.complex128))

    def forward(self, field):
        out = torch.matmul(field, self.weights)
        return out


class PTC(nn.Module):
    def __init__(self, input_dof, output_dof, path_list, gauge_field):
        super(PTC, self).__init__()
        self.PTC_layers = nn.ModuleList(
            [nn.Sequential(NonGaugeLinear(input_dof, output_dof), T(p, gauge_field)) for p
             in path_list])

    def forward(self, field):
        out = self.PTC_layers[0](field)
        for index in range(1, len(self.PTC_layers)):
            out += self.PTC_layers[index](field)
        return out

    def gauge_tra(self, new_gauge):
        for layer in self.PTC_layers:
            layer[1].gauge_tra(new_gauge)


class ComplexPE1(nn.Module):
    def __init__(self, gauge_field, non_gauge_dof, dupe_factor):
        super(ComplexPE1, self).__init__()
        lattice_size = gauge_field.shape[1:-2]
        volume = 1
        for dimension_size in lattice_size:
            volume *= dimension_size
        gauge_dof = gauge_field.shape[-1]
        dupe_size = non_gauge_dof * dupe_factor

        self.pe_tensor = torch.zeros(volume, gauge_dof, dupe_size, dtype=torch.complex128)
        for index_i in range(volume):
            for index_k in range(dupe_size):
                if index_k % 2 == 0:
                    self.pe_tensor[index_i][index_k % gauge_dof][index_k] = torch.sin(
                        torch.tensor(index_i / (10000 ** (index_k / (dupe_size - 2)))))
                if index_k % 2 == 1:
                    self.pe_tensor[index_i][index_k % gauge_dof][index_k] = 1j * torch.cos(
                        torch.tensor(index_i / (10000 ** ((index_k - 1) / (dupe_size - 2)))))
        self.pe_tensor = self.pe_tensor.reshape(*lattice_size, gauge_dof, dupe_size)

        self.dupe_tensor = torch.zeros(non_gauge_dof, dupe_size, dtype=torch.complex128)
        for index_j in range(dupe_size):
            self.dupe_tensor[index_j % non_gauge_dof][index_j] = 1

    def forward(self, field):
        out = torch.matmul(field, self.dupe_tensor)
        out = out + self.pe_tensor.unsqueeze(0)
        return out

    def gauge_tra(self, new_gauge):
        self.pe_tensor = gauge_tra(self.pe_tensor, new_gauge)


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
        x_block = torch.zeros(X_dim, gauge_dof, input_non_gauge_dof, dtype=torch.complex128)
        t_block = torch.zeros(T_dim, gauge_dof, input_non_gauge_dof, dtype=torch.complex128)
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

        self.dupe_tensor = torch.zeros(input_non_gauge_dof, 4 * input_non_gauge_dof, dtype=torch.complex128)
        for index_j in range(4 * input_non_gauge_dof):
            self.dupe_tensor[index_j % input_non_gauge_dof][index_j] = 1

        self.dominance = nn.Parameter(torch.tensor(2, dtype=torch.complex128), requires_grad=True)

    def forward(self, field):
        out = torch.matmul(field, self.dupe_tensor)
        out = ((out + self.pe_tensor.unsqueeze(0) * self.dominance) / self.dominance)
        return out

    def gauge_tra(self, new_gauge):
        self.pe_tensor = gauge_tra(self.pe_tensor, new_gauge)


class ComplexPE3(nn.Module):
    def __init__(self, gauge_field, non_gauge_dof, dupe_factor):
        super(ComplexPE3, self).__init__()
        lattice_size = gauge_field.shape[1:-2]
        volume = 1
        for dimension_size in lattice_size:
            volume *= dimension_size
        dupe_size = non_gauge_dof * dupe_factor

        self.pe_tensor = torch.zeros(volume, non_gauge_dof, dupe_size, dtype=torch.complex128)
        for index_i in range(volume):
            for index_k in range(dupe_size):
                if index_k % 2 == 0:
                    self.pe_tensor[index_i][index_k % non_gauge_dof][index_k] = torch.sin(
                        torch.tensor(index_i / (10000 ** (index_k / (dupe_size - 2)))))
                if index_k % 2 == 1:
                    self.pe_tensor[index_i][index_k % non_gauge_dof][index_k] = 1j * torch.cos(
                        torch.tensor(index_i / (10000 ** ((index_k - 1) / (dupe_size - 2)))))
        self.pe_tensor = self.pe_tensor.reshape(*lattice_size, non_gauge_dof, dupe_size)

    def forward(self, field):
        out = torch.matmul(field, self.pe_tensor)
        return out


class TraceActivation(nn.Module):
    def __init__(self, volume, gauge_field):
        super(TraceActivation, self).__init__()
        self.volume = volume
        temp = make_super_gauge_field(gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, *gauge_field.shape[-2:])
        self.activation = nn.Softmax(dim=-1)

    def forward(self, attention):
        matrix_values = torch.matmul(attention, dagger(attention))
        matrix_values = torch.abs(torch.sum(torch.diagonal(matrix_values, dim1=-2, dim2=-1), dim=-1)) / attention.shape[
            -1]
        effect = self.activation(matrix_values) / (matrix_values / 3)
        out = torch.einsum('Nhnmij,Nhnm->Nhnmij', [attention, effect])
        return out

    def gauge_tra(self, tra_gauge_field):
        temp = make_super_gauge_field(tra_gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, tra_gauge_field.shape[-2],
                                              tra_gauge_field.shape[-1])


class DeterminantActivation(nn.Module):
    def __init__(self, volume, gauge_field):
        super(DeterminantActivation, self).__init__()
        self.volume = volume
        temp = make_super_gauge_field(gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, *gauge_field.shape[-2:])
        self.activation = nn.Softmax(dim=-1)

    def forward(self, attention):
        matrix_values = torch.det(attention)
        matrix_values = torch.abs(matrix_values) ** (1 / attention.shape[-1])
        effect = self.activation(matrix_values)
        out = torch.einsum('nmij,Nhnm->Nhnmij', [self.super_gauge_field, effect])
        return out

    def gauge_tra(self, tra_gauge_field):
        temp = make_super_gauge_field(tra_gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, tra_gauge_field.shape[-2],
                                              tra_gauge_field.shape[-1])


class ComplexMatrixActivation(nn.Module):
    def __init__(self):
        super(ComplexMatrixActivation, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, energy):
        out = torch.sum(energy, dim=(-2, -1))
        out = torch.real(out) + torch.imag(out)
        out = self.softmax(out)
        out = out.to(torch.complex128)
        return out


class SelfAttention2(nn.Module):
    def __init__(self, gauge_field, heads):
        super(SelfAttention2, self).__init__()
        lattice = gauge_field.shape[1:-2]
        self.volume = 1
        for dim_size in lattice:
            self.volume *= dim_size

        self.gauge_field = gauge_field
        temp = make_super_gauge_field(self.gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, self.gauge_field.shape[-2],
                                              self.gauge_field.shape[-1])
        self.heads = heads
        self.activation = ComplexMatrixActivation()

    def forward(self, queries, keys, values):
        queries_shape = queries.shape
        keys_shape = keys.shape
        values_shape = values.shape
        queries = queries.reshape(queries_shape[0], self.volume, queries_shape[-2], -1, self.heads).permute(0, 4, 1, 2,
                                                                                                            3)
        keys = keys.reshape(keys_shape[0], self.volume, keys_shape[-2], -1, self.heads).permute(0, 4, 1, 2, 3)
        values = values.reshape(values_shape[0], self.volume, values_shape[-2], -1, self.heads).permute(0, 4, 1, 2, 3)
        norm = (queries.shape[-2] * queries.shape[-1]) ** (1 / 2)

        # start_time = time.time()
        energy = torch.einsum("nmis,NHmsj->NHnmij", [self.super_gauge_field, keys])
        energy = torch.einsum("NHnis,NHnmsj->NHnmij", [dagger(queries), energy])
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Energy in: {execution_time*1e3:.3f} ms")

        # start_time = time.time()
        attention = self.activation(energy / norm)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Attention Matrix in: {execution_time*1e3:.3f} ms")

        # start_time = time.time()
        out = torch.einsum("nmis,NHmsj->NHnmij", [self.super_gauge_field, values])
        out = torch.einsum("NHnmij,NHnm->NHnij", [out, attention])

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"SelfAttention in: {execution_time*1e3:.3f} ms")
        out = out.permute(0, 2, 3, 4, 1).reshape(*queries_shape)
        return out

    def gauge_tra(self, new_gauge):
        self.gauge_field = gauge_tra(self.gauge_field, new_gauge, field_is_gauge_field=True)
        temp = make_super_gauge_field(self.gauge_field)
        self.super_gauge_field = temp.reshape(self.volume, self.volume, self.gauge_field.shape[-2],
                                              self.gauge_field.shape[-1])


class SelfAttention(nn.Module):
    def __init__(self, gauge_field, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.gauge_field = gauge_field
        lattice = gauge_field.shape[1:-2]
        self.volume = 1
        for dim_size in lattice:
            self.volume *= dim_size
        self.activation = TraceActivation(self.volume, self.gauge_field)

    def forward(self, queries, keys, values):
        queries_shape = queries.shape
        keys_shape = keys.shape
        values_shape = values.shape
        queries = queries.reshape(queries_shape[0], self.volume, queries_shape[-2], -1, self.heads).permute(0, 4, 1, 2,
                                                                                                            3)
        keys = keys.reshape(keys_shape[0], self.volume, keys_shape[-2], -1, self.heads).permute(0, 4, 1, 2, 3)
        values = values.reshape(values_shape[0], self.volume, values_shape[-2], -1, self.heads).permute(0, 4, 1, 2, 3)

        attention = torch.einsum("Nhnis,Nhmjs->Nhnmij", [queries, torch.conj(keys)])
        attention = self.activation(attention / queries.shape[-1])
        out = torch.einsum("Nhnmis,Nhmsj->Nhnij", [attention, values])
        #print("values ", torch.mean(torch.abs(values)))
        #print("attention ", torch.mean(torch.abs(out)))
        out = out.permute(0, 2, 3, 4, 1).reshape(*values_shape[:-1], -1)
        return out
       
    def gauge_tra(self, new_gauge):
        self.gauge_field = gauge_tra(self.gauge_field, new_gauge, field_is_gauge_field=True)
        self.activation.gauge_tra(self.gauge_field)


class Transformer(nn.Module):
    def __init__(self, gauge_field, input_non_gauge_dof, heads, head_dim):
        super(Transformer, self).__init__()
        lattice = gauge_field.shape[1:-2]

        assert head_dim % len(lattice) == 0, "Head - dimension must be divisible by number of lattice - dimensions"

        embedding_size = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim

        self.pe = PE_4D(gauge_field, input_non_gauge_dof)

        self.W_Q = ReducedNonGaugeLinear(input_non_gauge_dof, embedding_size, lattice)
        self.W_K = ReducedNonGaugeLinear(input_non_gauge_dof, embedding_size, lattice)
        self.W_V = NonGaugeLinear(input_non_gauge_dof, embedding_size)

        self.self_attention = SelfAttention(gauge_field, heads)
        self.out = NonGaugeLinear(embedding_size, input_non_gauge_dof)

    def forward(self, field):
        # start_time = time.time()
        queries = self.W_Q(self.pe(field))
        keys = self.W_K(self.pe(field))
        values = self.W_V(field)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Q,K,V in: {execution_time*1e3:.3f} ms")
        # print("Values Mean: ", torch.mean(torch.abs(values)))
        out = self.self_attention(queries, keys, values)
        # print("SA mean: ", torch.mean(torch.abs(out)))
        out = self.out(out)
        #print(torch.mean(torch.abs(out)))
        return out

    def gauge_tra(self, new_gauge):
        self.pe.gauge_tra(new_gauge)
        self.self_attention.gauge_tra(new_gauge)


class PtcSelfAttention(nn.Module):
    def __init__(self, gauge_field):
        super(PtcSelfAttention, self).__init__()
        self.gauge_field = gauge_field
        lattice = gauge_field.shape[1:-2]
        self.volume = 1
        for dim_size in lattice:
            self.volume *= dim_size
        self.activation = TraceActivation(self.volume, self.gauge_field)

    def forward(self, queries, keys, values):
        queries_shape = queries.shape
        keys_shape = keys.shape
        values_shape = values.shape
        queries = queries.reshape(queries_shape[0], self.volume, *queries_shape[-2:])
        keys = keys.reshape(keys_shape[0], self.volume, *keys_shape[-2:])
        values = values.reshape(values_shape[0], self.volume, *values_shape[-2:])

        attention = torch.einsum("Nnis,Nmjs->Nnmij", [queries, torch.conj(keys)])
        attention = self.activation(attention / queries.shape[-1] ** 0.5)
        out = torch.einsum("Nnmis,Nmsj->Nnij", [attention, values])

        out = out.reshape(*values_shape)
        return out

    def gauge_tra(self, new_gauge):
        self.gauge_field = gauge_tra(self.gauge_field, new_gauge, field_is_gauge_field=True)
        self.activation.gauge_tra(self.gauge_field)


class PtcTransformer(nn.Module):
    def __init__(self, input_non_gauge_dof, ptc_paths, gauge_field):
        super(PtcTransformer, self).__init__()

        self.W_Q = PTC(input_non_gauge_dof, input_non_gauge_dof, ptc_paths, gauge_field)
        self.W_K = PTC(input_non_gauge_dof, input_non_gauge_dof, ptc_paths, gauge_field)
        self.W_V = PTC(input_non_gauge_dof, input_non_gauge_dof, ptc_paths, gauge_field)

        self.self_attention = PtcSelfAttention(gauge_field)

    def forward(self, field):
        # start_time = time.time()
        queries = self.W_Q(field)
        keys = self.W_K(field)
        values = self.W_V(field)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Q,K,V in: {execution_time*1e3:.3f} ms")
        # print("Values Mean: ", torch.mean(torch.abs(values)))
        out = self.self_attention(queries, keys, values)
        # print("SA mean: ", torch.mean(torch.abs(out)))
        return out

    def gauge_tra(self, new_gauge):
        self.W_Q.gauge_tra(new_gauge)
        self.W_K.gauge_tra(new_gauge)
        self.W_V.gauge_tra(new_gauge)
        self.self_attention.gauge_tra(new_gauge)
