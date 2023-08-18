import numpy as np

from Fisher_KPP_Domain import *
from PDDO_STRidge import *
from PDDO_Utilities import *
# Loading the dataset, which consists of coordinates, field variable
# values, level sets and interface coordinates
ucs = np.load("dataset/Coords_xy.npy", allow_pickle=True)
uvals = np.load("dataset/TVals_xy.npy", allow_pickle=True)
phivals = np.load("dataset/PhiVals_xy.npy", allow_pickle=True)
numtime = 101
delta = 4.015
dx = 10.0 / 400
dt = 0.01
# Creating the field object for given experiment
field = DM2D(numtime=numtime, delta=delta, dx=dx, dt=dt, varname="u")

# Check if the families are defined
field.check_fam_build()
# Split the field to get the occupied region
field.split_fam_gen(ucs, uvals, phivals[0])
# Creating the list consisting the derivative terms
der_list, u_ders = field.gen_derlist(order=2)
noise_lev = 0.0

# Calculating the derivatives using PDDO
dmat = field.gendmat(2)
for i in range(numtime - 2):
    print(i)
    # Adding the noise to field variable
    rmse = np.std(field.u_vals[i])
    field.u_vals[i] += np.random.normal(0, noise_lev * rmse / 100.0, len(field.u_vals[i]))
    for k in range(len(der_list)):
        u_ders[k].append(dmat[k].dot(field.u_vals[i]))

# Saving the derivatives and velocities
np.save('uders_car.npy', u_ders, allow_pickle=True)
np.save('uderlist_car.npy', der_list, allow_pickle=True)
np.save('vels_field.npy', field.velocity, allow_pickle=True)

# Term combinations and operators for the feature matrix creation
term_comb = [[0], [0, 0], [3, 5], [1], [2], [3], [4], [5], [2, 3]]
operators = [[], ['*'], ['+'], [], [], [], [], [], ['*']]
# Loading the velocities and field variable derivatives
u_ders = np.load("uders_car.npy", allow_pickle=True)
vels = np.load("vels_field.npy", allow_pickle=True)

bagging_learn = BGEL(u_ders=u_ders, vels=vels, lambda_2=1.0, numtime=numtime-2, model_name='field')
# Bootstrapping the data for given parameters
bagging_learn.bootstrap_data_lib(60, 9, 10)
# Creating the feature matrix
bagging_learn.crt_can_lib(term_comb, operators, der_list)
# Applying Ensemble SINDy
bagging_learn.bagg_learn()
# Calculating the inclusion probabilities
bagging_learn.calc_inc_prob(0.7)
# Calculating the median and standard deviation values of coefficients
bagging_learn.calc_med_std()
# Plotting the coefficient discovery results
bagging_learn.plt_coeff_dist()