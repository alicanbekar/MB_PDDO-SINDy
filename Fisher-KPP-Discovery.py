import numpy as np

from fisher_kpp_domain_3 import *
from PDDO_STRidge import *
from PDDO_Utilities import *

ucs = np.load("dataset/Coords_xy.npy", allow_pickle=True)
uvals = np.load("dataset/TVals_xy.npy", allow_pickle=True)
phivals = np.load("dataset/PhiVals_xy.npy", allow_pickle=True)

field = DM2D(numtime=100, delta=4.015, dx=20.0/200, dt=0.4, varname="u")
"""

field.check_fam_build()
field.split_fam_gen(ucs, uvals, phivals[0])
"""
der_list, u_ders = field.gen_derlist(order=2)
noise_lev = 0.0
"""

dmat = field.gendmat(2)
for i in range(100-1):
    print(i)
    rmse = np.std(field.u_vals[i])
    print(rmse)
    field.u_vals[i] += np.random.normal(0, noise_lev * rmse / 100.0, len(field.u_vals[i]))
    for k in range(len(der_list)):
        u_ders[k].append(dmat[k].dot(field.u_vals[i]))

np.save('uders_car.npy', u_ders, allow_pickle=True)
np.save('uderlist_car.npy', der_list, allow_pickle=True)
np.save('vels_field.npy', field.velocity, allow_pickle=True)
"""
term_comb = [[0], [0, 0], [3, 5], [1], [2], [3], [4], [5], [2, 3]]
operators = [[], ['*'], ['+'], [], [], [], [], [], ['*']]
u_ders = np.load("uders_car.npy", allow_pickle=True)
vels = np.load("vels_field.npy", allow_pickle=True)

bagging_learn = BGEL(u_ders=u_ders, vels=vels, lambda_2=1.0, numtime=99, model_name='field')
bagging_learn.bootstrap_data_lib(99, 8, 10)
bagging_learn.crt_can_lib(term_comb, operators, der_list)
bagging_learn.bagg_learn()
bagging_learn.calc_inc_prob(0.7)
bagging_learn.calc_med_std()
bagging_learn.plt_coeff_dist()

# dr_learn = DRL(numtime=100, num_feat=8, model_name='field')
# dr_learn.crt_can_lib(term_comb, operators, der_list)
# dr_learn.crt_feat_mat_velvec(u_ders, vels)
# dr_learn.normalize_system()
# dr_learn.doug_rach(0.2, 1.0, 0.5, 1000, 1e-12)
