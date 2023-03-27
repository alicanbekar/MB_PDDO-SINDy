import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import shape, std
from scipy import spatial
from scipy.fftpack import rfft, irfft, fftfreq, fft
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import mean_squared_error
from PDDO_Utilities import *
import matplotlib.pyplot as plt

from itertools import compress
import random

random.seed(42)
print(random.random())

class DM2D:
    """
    
    """
    def __init__(self, numtime, delta, dx, dt, varname):
        self.numtime = numtime
        self.delta = delta
        self.dx = dx
        self.dt = dt
        self.varname = varname
        self.velocity = []
        self.u_coords = []
        self.u_vals = []
        self.fams = []
        self.samp_ids = []
        self.u_samp = []
        self.fam_exst = False

    def check_fam_build(self):
        print(self.fam_exst)


    def split_fam_gen(self, d_coords, d_vals, init_lev_sets):
        self.fam_exst = True
        bool_reg = (init_lev_sets < 0)
        self.u_coords = list(compress(d_coords, bool_reg))
        tree_u = spatial.KDTree(self.u_coords)
        for k in self.u_coords:
            fam_pts = tree_u.query_ball_point(
                k, self.delta * self.dx, return_sorted=True
            )
            self.fams.append(fam_pts)
        for i in range(self.numtime - 1):
            self.u_vals.append(list(compress(d_vals[i], bool_reg)))
            self.velocity.append(list(compress((d_vals[i + 1] - d_vals[i]) / self.dt, bool_reg)))


    def gendmat(self, order):
        nonzcounter = 0
        lenxy = len(self.u_coords)
        totnz = sum([len(listElem) for listElem in self.fams])
        _, _, _, lens = PD2D.binomial_coeff(order + 1)
        dmats = []
        dmatscsr = []
        amat = np.zeros([lens, lens])
        for _ in range(lens):
            dmats.append(
                [
                    np.zeros(totnz, dtype=np.int),
                    np.zeros(totnz, dtype=np.int),
                    np.zeros(totnz),
                ]
            )
        for i in range(lenxy):
            bmat = PD2D.gen_bmat(order)
            pts = self.fams[i]
            amat.fill(0.0)
            for k in pts:
                xi = self.u_coords[k] - self.u_coords[i]
                amat += PD2D.gen_amat(xi, self.delta * self.dx, order)
            amat *= self.dx * self.dx
            cfs = np.linalg.solve(amat, bmat)
            for k in pts:
                xi = self.u_coords[k] - self.u_coords[i]
                gfs = PD2D.inv_gfunc(xi, self.delta * self.dx, cfs, order)
                for derord in range(lens):
                    dmats[derord][0][nonzcounter] = i
                    dmats[derord][1][nonzcounter] = k
                    dmats[derord][2][nonzcounter] = gfs[derord] * self.dx * self.dx
                nonzcounter += 1
        for i in range(lens):
            dmatscsr.append(
                csr_matrix(
                    (dmats[i][2], (dmats[i][0], dmats[i][1])),
                    shape=(lenxy, lenxy),
                )
            )
        return dmatscsr

    def gen_derlist(self, order):
        _, ord1, ord2, _ = PD2D.binomial_coeff(order + 1)
        ord1 = [ord * "x" for ord in ord1]
        ord2 = [ord * "y" for ord in ord2]
        der_list = [self.varname + "_{" + i + j + "}" for i, j in zip(ord1, ord2)]
        der_list[0] = self.varname
        u_ders = [[] for x in range((order + 1) * (order + 2) // 2)]
        return der_list, u_ders
