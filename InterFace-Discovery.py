import numpy as np
from moving_boundary_curve import *
from PDDO_STRidge import *
from PDDO_Utilities import *

ucs = np.load("dataset/Coords_xy.npy", allow_pickle=True)
uvals = np.load("dataset/TVals_xy.npy", allow_pickle=True)
phivals = np.load("dataset/PhiVals_xy.npy", allow_pickle=True)
intercs = np.load("dataset/interface_coords.npy", allow_pickle=True)


# ucs = np.load("u_coordinates.npy", allow_pickle=True)
# intercs = np.load("interface_coordinates.npy", allow_pickle=True)
# uvals = np.load("uvals.npy", allow_pickle=True)
# phivals = np.load("phivals.npy", allow_pickle=True)
mov_boundary = MB2D(
    numtime=100, delta=4.015, dx=20.0/200, dt=0.4, varname="u", u_coords=ucs
)
mov_boundary.calc_normtan(intercs, 0.0)
mov_boundary.fam_gen(phivals)
mov_boundary.calc_vel()
#mov_boundary.correct_normtan_ls(phivals)
der_list, u_ders = mov_boundary.gen_derlist(order=2)
noise_lev = 5

"""
"""
for i in range(100-1):
    print(i)
    rmse = mean_squared_error(uvals[i], np.zeros(uvals[i].shape), squared=False)
    uvals[i] += np.random.normal(0, noise_lev * rmse / 100.0, uvals[i].shape)
    dmat = mov_boundary.gendmat(2, i)
    for k in range(len(der_list)):
        u_ders[k].append(dmat[k].dot(uvals[i]))

np.save('uders.npy', u_ders, allow_pickle=True)
np.save('uderlist.npy', der_list, allow_pickle=True)
np.save('vels.npy', mov_boundary.velocity, allow_pickle=True)

term_comb = [[0], [1], [2], [1, 1], [1, 2], [2, 2], [3], [5], [4], [4, 5]]
operators = [[], [], [], ['*'], ['*'], ['*'], [], [], [], ['+']]
u_ders = np.load("uders.npy", allow_pickle=True)
vels = np.load("vels.npy", allow_pickle=True)

bagging_learn = BGEL(u_ders=u_ders, vels=vels, lambda_2=1.0, numtime=99, model_name='moving_boundary')
bagging_learn.bootstrap_data_lib(80, 9, 11)
bagging_learn.crt_can_lib(term_comb, operators, der_list)
bagging_learn.bagg_learn()
bagging_learn.calc_inc_prob(0.7)
bagging_learn.calc_med_std()
bagging_learn.plt_coeff_dist()
