import numpy as np
from Moving_Boundary_Curve import *
from PDDO_STRidge import *
from PDDO_Utilities import *

# Loading the dataset, which consists of coordinates, field variable
# values, level sets and interface coordinates
ucs = np.load("dataset/Coords_xy.npy", allow_pickle=True)
uvals = np.load("dataset/TVals_xy.npy", allow_pickle=True)
phivals = np.load("dataset/PhiVals_xy.npy", allow_pickle=True)
intercs = np.load("dataset/interface_coords.npy", allow_pickle=True)
numtime = 101
delta = 4.015
dx = 10.0 / 400
dt = 0.01
# Creating the moving boundary object for given experiment
mov_boundary = MB2D(
    numtime=numtime, delta=delta, dx=dx, dt=dt, varname="u", u_coords=ucs
)
# Calculating the normal and tangential directions of sensor points on the
# moving boundary
mov_boundary.calc_normtan(intercs, 0.0)
# Creating the families around the moving boundary sensor locations for
# field variable derivative calculation
mov_boundary.fam_gen(phivals)
# Calculating the moving boundary velocities in normal direction
mov_boundary.calc_vel()
# Level set correction for the normal and tangential directions default is off
# mov_boundary.correct_normtan_ls(phivals)
# Creating the list consisting the derivative terms
der_list, u_ders = mov_boundary.gen_derlist(order=2)
# Amount of Gaussian noise in field variable, x%
noise_lev = 1

# Calculating the derivatives using PDDO
for i in range(numtime - 2):
    print(i)
    # Adding the noise to field variable
    rmse = mean_squared_error(uvals[i], np.zeros(uvals[i].shape), squared=False)
    uvals[i] += np.random.normal(0, noise_lev * rmse / 100.0, uvals[i].shape)
    # Matrix operator for derivative calculation
    dmat = mov_boundary.gendmat(2, i)
    for k in range(len(der_list)):
        u_ders[k].append(dmat[k].dot(uvals[i]))

# Saving the derivatives and velocities
np.save("uders.npy", u_ders, allow_pickle=True)
np.save("uderlist.npy", der_list, allow_pickle=True)
np.save("vels.npy", mov_boundary.velocity, allow_pickle=True)

# Term combinations and operators for the feature matrix creation
term_comb = [[0], [1], [2], [1, 1], [1, 2], [2, 2], [3], [5], [4], [4, 5]]
operators = [[], [], [], ["*"], ["*"], ["*"], [], [], [], ["+"]]
# Loading the velocities and field variable derivatives
u_ders = np.load("uders.npy", allow_pickle=True)
vels = np.load("vels.npy", allow_pickle=True)

# Creating the Ensemble SINDy class for given data and experiment setup
bagging_learn = BGEL(
    u_ders=u_ders,
    vels=vels,
    lambda_2=1.0,
    numtime=numtime - 2,
    model_name="moving_boundary",
)
# Bootstrapping the data for given parameters
bagging_learn.bootstrap_data_lib(60, 8, 11)
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
