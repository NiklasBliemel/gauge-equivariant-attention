from GMRES import gmres
from Operators import D_WC
from Constants import *
from Preconditioner import transformer, ptc
import pickle
from time import perf_counter_ns

d_wc = D_WC(M, GAUGE_FIELD)


def safe_gmres(tra_name):
    
    preconditioner = transformer(tra_name)

    b = torch.rand(NUM_OF_LATTICES, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)

    time_before = perf_counter_ns()
    field, res, steps, res_list = gmres(d_wc, b, preconditioner, preconditioner(b), 1000, 0.01, True)
    time_taken = (perf_counter_ns() - time_before) * 1e-6
    error = torch.norm((d_wc(field) - b).view(-1))
    res_list.append(time_taken)
    res_list.append(error)

    torch.save(field, "hpd_test/" + tra_name + ".pt")

    with open("hpd_test/" + tra_name + ".pkl", 'wb') as f:
        pickle.dump(res_list, f)
        
safe_gmres("Tr_4_16_True")
