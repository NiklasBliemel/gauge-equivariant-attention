from GMRES import gmres
from Operators import D_WC
from Constants import *
from Preconditioner import transformer, ptc
import pickle
from time import perf_counter_ns

d_wc = D_WC(M, GAUGE_FIELD_SMALL)

Preconditioner = transformer("tr_4_16_True")

b = torch.rand(NUM_OF_LATTICES, *LATTICE_SMALL, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)

time_before = perf_counter_ns()
field, res, steps, res_list = gmres(d_wc, b, Preconditioner, Preconditioner(b), 1000, 0.01, True)
time_taken = (perf_counter_ns() - time_before) * 1e-6
error = torch.norm((d_wc(field) - b).view(-1))

torch.save(field, "hpd_test/solution_tra_0.pt")

with open('hpd_test/res_list_tra_0.pkl', 'wb') as f:
    pickle.dump(res_list, f)
