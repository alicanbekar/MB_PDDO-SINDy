from math import atan2
from itertools import compress
from scipy.sparse import csr_matrix
from scipy import spatial
from sklearn.metrics import mean_squared_error
import numpy as np


from PDDO_Utilities import *


class MB2D:
    """
    The class to represent 2D moving boundary curves.
    """

    def __init__(self, numtime, delta, dx, dt, varname, u_coords):
        self.numtime = numtime
        self.delta = delta
        self.dx = dx
        self.dt = dt
        self.varname = varname
        self.u_coords = u_coords
        self.n_angles = []
        self.t_angles = []
        self.inter_cs = []
        self.inter_fams = []
        self.velocity = []

    def calc_normtan(self, intercs, noise_perc_inter):
        """
        Generates tangent and normal angles between the given curve
        and x axis. It interpolates the given set of coordinate points
        in 2D and uses atan2 function to find the tangent.
        """
        self.noise_inter = noise_perc_inter
        for i in range(self.numtime):
            #intercs[i] += np.random.normal(0, self.noise_inter * self.dx / 100.0, intercs[i].shape)
            n_interpts = len(intercs[i])
            self.t_angles.append(np.zeros((n_interpts - 1)))
            self.n_angles.append(np.zeros((n_interpts - 1)))
            self.inter_cs.append(np.zeros_like(intercs[i], shape=[n_interpts - 1, 2]))
            for k in range(n_interpts - 1):
                inc = (intercs[i][k + 1] - intercs[i][k]) / 2.0
                self.inter_cs[i][k] = intercs[i][k] + inc
                self.t_angles[i][k] = atan2(inc[1], inc[0])
                self.n_angles[i][k] = atan2(inc[1], inc[0]) - np.pi / 2.0

    def fam_gen(self, lev_sets):
        """
        Generates family member indices for given timeseries
        coordinates for 2D curves. The family members are from
        two separate regions which the given curve separates.
        For family the process of family generation go to:
        Ref:[Madenci, E., Barut, A., & Futch, M. (2016). ScienceDirect
        Peridynamic differential operator and its applications]
        """
        tree_u = spatial.KDTree(self.u_coords)
        for i in range(self.numtime):
            self.inter_fams.append([])
            n_interpts = len(self.inter_cs[i])
            for k in range(n_interpts):
                sym_pts = tree_u.query_ball_point(
                    self.inter_cs[i][k], self.delta * self.dx, return_sorted=True
                )
                bool_reg = lev_sets[i][sym_pts] <= 0.0
                self.inter_fams[i].append(list(compress(sym_pts, bool_reg)))

    def correct_normtan_ls(self, lev_sets):
        """
        Normal and tangent direction corrections using level-set values.
        """
        order = 1
        _, _, _, lens = PD2D.binomial_coeff(order + 1)
        amat = np.zeros([lens, lens])
        for i in range(self.numtime):
            #lev_sets[i] += np.random.normal(0, self.noise_inter * self.dx / 100.0, lev_sets[i].shape)
            n_interpts = len(self.inter_cs[i])
            for k in range(n_interpts):
                bmat = PD2D.gen_bmat(order)
                pts = self.inter_fams[i][k]
                amat.fill(0.0)
                for num in self.u_coords[pts]:
                    xi = num - self.inter_cs[i][k]
                    amat += PD2D.gen_amat(xi.ravel(), self.delta * self.dx, order)
                amat *= self.dx * self.dx
                cfs = np.linalg.solve(amat, bmat)
                n_x = 0.0
                n_y = 0.0
                for j in pts:
                    xi = self.u_coords[j] - self.inter_cs[i][k]
                    gfs = PD2D.inv_gfunc(xi.ravel(), self.delta * self.dx, cfs, order)
                    n_x += gfs[1] * lev_sets[i][j] * self.dx * self.dx
                    n_y += gfs[2] * lev_sets[i][j] * self.dx * self.dx
                self.n_angles[i][k] = atan2(n_y, n_x)
                self.t_angles[i][k] = atan2(n_y, n_x) + np.pi / 2.0

    def calc_vel(self):
        """
        Calculates the velocity of the given 2D curve to its normal
        direction. For every point on the curve, the closest point from the
        next timestamp is found and the vector distance between these two
        points is dotted with the curve normal at the point of interest.
        """
        for i in range(self.numtime - 1):
            n_interpts = len(self.inter_cs[i])
            self.velocity.append(np.zeros((n_interpts)))
            tree = spatial.KDTree(self.inter_cs[i + 1])
            for k in range(n_interpts):
                n_x = np.cos(self.n_angles[i][k])
                n_y = np.sin(self.n_angles[i][k])
                ptcoord = self.inter_cs[i][k]
                _, nghpt = tree.query(ptcoord, k=1)
                distvec = self.inter_cs[i + 1][nghpt] - ptcoord
                self.velocity[i][k] = (distvec[0] * n_x + distvec[1] * n_y) / self.dt

    def gendmat(self, order, ts):
        """
        Global derivative matrix calculation using PDDO.
        """
        nonzcounter = 0
        leninter = len(self.inter_cs[ts])
        lenreg = len(self.u_coords)
        totnz = sum([len(listElem) for listElem in self.inter_fams[ts]])
        _, _, _, lens = PD2D.binomial_coeff(order + 1)
        dmats = []
        dmatscsr = []
        amat = np.zeros([lens, lens])
        for i in range(lens):
            dmats.append(
                [
                    np.zeros(totnz, dtype=np.int),
                    np.zeros(totnz, dtype=np.int),
                    np.zeros(totnz),
                ]
            )
        for i in range(leninter):
            ct = np.cos(self.t_angles[ts][i])
            st = np.sin(self.t_angles[ts][i])
            c = np.array([[ct, st], [-st, ct]])
            bmat = PD2D.gen_bmat(order)
            pts = self.inter_fams[ts][i]
            amat.fill(0.0)
            for num in self.u_coords[pts]:
                xi = num - self.inter_cs[ts][i]
                xirot = np.dot(c, xi.T).T
                amat += PD2D.gen_amat(xirot.ravel(), self.delta * self.dx, order)
            amat *= self.dx * self.dx
            cfs = np.linalg.solve(amat, bmat)
            for k in pts:
                xi = self.u_coords[k] - self.inter_cs[ts][i]
                xirot = np.dot(c, xi.T).T
                gfs = PD2D.inv_gfunc(xirot.ravel(), self.delta * self.dx, cfs, order)
                # Filling the sparse matrices for different order derivative
                # calculations.
                for derord in range(lens):
                    dmats[derord][0][nonzcounter] = i
                    dmats[derord][1][nonzcounter] = k
                    dmats[derord][2][nonzcounter] = gfs[derord] * self.dx * self.dx
                nonzcounter += 1
        for i in range(lens):
            dmatscsr.append(
                csr_matrix(
                    (dmats[i][2], (dmats[i][0], dmats[i][1])), shape=(leninter, lenreg)
                )
            )
        return dmatscsr

    def gen_derlist(self, order):
        _, ord1, ord2, _ = PD2D.binomial_coeff(order + 1)
        ord1 = [ord * "x_t" for ord in ord1]
        ord2 = [ord * "x_n" for ord in ord2]
        der_list = [self.varname + "_{" + i + j + "}" for i, j in zip(ord1, ord2)]
        der_list[0] = self.varname
        u_ders = [[] for x in range((order + 1) * (order + 2) // 2)]
        return der_list, u_ders