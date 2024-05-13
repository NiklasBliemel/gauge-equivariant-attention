from GMRES import GmresTest
from Operators import D_WC
from Constants import *
from Preconditioner import *

d_wc = D_WC(M, GAUGE_FIELD)

gmres_test = GmresTest(d_wc, NUM_OF_LATTICES, LATTICE, GAUGE_DOF, NON_GAUGE_DOF, max_iter=100, tol=1e-1)

Tr = transformer("Tr_4_16_True")
gmres_test(Tr)
