from modules.GMRES import gmres
from modules.Operators import D_WC
from modules.Constants import *
from modules.Preconditioner import transformer
import pickle
import time

d_wc = D_WC(M, GAUGE_FIELD)
d_wc_small = D_WC(M, GAUGE_FIELD_SMALL)


def safe_gmres(tra_name, small=False):
    if small:
        operator = d_wc_small
        b = torch.rand(NUM_OF_LATTICES, *LATTICE_SMALL, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)
    else:
        operator = d_wc
        b = torch.rand(NUM_OF_LATTICES, *LATTICE, GAUGE_DOF, NON_GAUGE_DOF, dtype=torch.complex64)

    preconditioner = transformer(tra_name)
    time_before = time.time()
    field, res, steps, res_list = gmres(operator, b, preconditioner, preconditioner(b), 1000, 0.01, True)
    time_taken = (time.time() - time_before) * 1e3
    error = torch.norm((operator(field) - b).view(-1))
    res_list.append(time_taken)
    res_list.append(error)

    torch.save(field, "hpd_test/" + tra_name + ".pt")

    with open("hpd_test/" + tra_name + ".pkl", 'wb') as f:
        pickle.dump(res_list, f)


safe_gmres("tr_gmres_4_16_True", small=True)
