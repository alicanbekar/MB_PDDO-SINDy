import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import shape, std
from scipy import spatial
from scipy.fftpack import rfft, irfft, fftfreq, fft
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import mean_squared_error
from PDDO_Utilities import *

from itertools import compress


def FamGen(uCs, delta, dx, Phis, Tvals, numtstp, delt):
    """Generates family member indices for given timeseries
    coordinates for the domain variable. The family members
    are from two separate regions which the given curve separates.
    For family the process of family generation go to:
    Ref:[Madenci, E., Barut, A., & Futch, M. (2016). ScienceDirect
    Peridynamic differential operator and its applications]

    Inputs
    ------
    u1Cs : list of lists for domain coordinates (doubles).
    Shape = [numtime][numcoord_u(t)][2].

    delta : double. Horizon radius or radius of search for family
    members as a multiplier of dx.

    dx : double. Mesh size. The given mesh for the domains is assumed
    to be constant and dx=dy.

    numtstp : integer. Number of timesteps.

    Outputs
    -------
    InterFams : list of list of lists (integers). Shape = [numtstp][numcoords(t)]
    It takes family member indices for surrounding region of the curve.
    """
    Fams = []
    Coords = []
    Ts = []
    Vels = []
    for i in range(numtstp):
        print(i)
        Domainpts = Phis[i] <= 0.0
        Coords.append(list(compress(uCs, Domainpts)))
        Ts.append(list(compress(Tvals[i], Domainpts)))
        Vels.append(list(compress((Tvals[i + 1] - Tvals[i]) / delt, Domainpts)))
        tree_u = spatial.KDTree(Coords[i])
        # Append two list per timestep for family members in two domains.
        Fams.append([])
        # Create and object tree for domain coordinates for ith timestep
        # for faster family search.
        PtsNum = len(Coords[i])
        for k in range(PtsNum):
            # Search, sort and append the material points inside the horizon.
            SymPoints = tree_u.query_ball_point(
                Coords[i][k], delta * dx, return_sorted=True
            )
            Fams[i].append(SymPoints)
    return Fams, Coords, Ts, Vels


# Derivative Operator Using PDDO
def GenDmat(dx, delta, order, Fams, Coords, Varname):
    nonzcounter = 0
    lenxy = len(Coords)
    totnz = sum([len(listElem) for listElem in Fams])
    _, ord1, ord2, lens = BinomialCoeff(order + 1)
    DerList = GenDerList(ord1, ord2)
    Dmats = []
    DmatsCsc = []
    Amat = np.zeros([lens, lens])
    for i in range(lens):
        Dmats.append(
            [
                np.zeros(totnz, dtype=np.int),
                np.zeros(totnz, dtype=np.int),
                np.zeros(totnz),
            ]
        )
    for i in range(lenxy):
        bmat = Genbmat(order)
        pts = Fams[i]
        Amat.fill(0.0)
        for num in pts:
            xi = Coords[num] - Coords[i]
            Amat += GenAmat(xi.ravel(), delta * dx, order)
        Amat *= dx * dx
        amat = np.linalg.solve(Amat, bmat)
        for k in pts:
            xi = Coords[k] - Coords[i]
            GFS = InvGFunc(xi.ravel(), delta * dx, amat, order)
            for derord in range(lens):
                Dmats[derord][0][nonzcounter] = i
                Dmats[derord][1][nonzcounter] = k
                Dmats[derord][2][nonzcounter] = GFS.T[derord] * dx * dx
            nonzcounter += 1
    for i in range(lens):
        DmatsCsc.append(
            csr_matrix((Dmats[i][2], (Dmats[i][0], Dmats[i][1])), shape=(lenxy, lenxy))
        )
    return DmatsCsc, DerList


def GenDerList(ordlist1, ordlist2):
    ord1_tag = list(map(str, ordlist1))
    ord2_tag = list(map(str, ordlist2))
    DerList = ["u" + i + j for i, j in zip(ord1_tag, ord2_tag)]
    return DerList


def FmatCrt(TDers_org, DerList, bootindx):
    """Creates feature matrix using the field derivatives on the
    moving interface. Data from both regions is included but products
    between two regions is not. Names of the features are also created.
    Feature matrix columns are created manually but they can be modified
    with a function which creates the combinational products of the
    field derivatives.

    Inputs
    ------
    T1Ders : array of doubles with shape = ((order + 1) * (order + 2) // 2,
    numtime, numcoord(t)). It consists of the derivatives of the
    dependend variable from the 1st region of the domain in increasing
    order on the interface coordinates sequentially.

    T2Ders :  array of doubles with shape = ((order + 1) * (order + 2) // 2,
    numtime, numcoord(t)). It consists of the derivatives of the
    dependend variable from the 2nd region of the domain in increasing
    order on the interface coordinates sequentially.

    Derlist1 : list of strings. Includes the names of the derivatives for
    the 1st region.

    Derlist2 : list of strings. Includes the names of the derivatives for
    the 2nd region.

    bootindx :list of arrays of doubles. Shape = [numtime](numcoord(t)).
    It contains the time derivative values of the interface in normal
    direction.

    Outputs
    -------
    Fmat : array of doubles with shape = (numfeat,). Contains the coefficients
    of the regression.
    FSList : list of strings with shape = [numfeat]. It contains the tagnames
    for the features.
    """
    TDers = [[] for x in range(len(DerList))]
    for k1 in range(len(DerList)):
        TDers[k1] = np.concatenate(TDers_org[k1][bootindx[:]]).ravel()
    bias = np.ones_like(TDers[0])
    numfeat = 9
    totfeat = numfeat + 1
    Fmat = [[] for x in range(totfeat)]
    FSList = [[] for x in range(totfeat)]
    Fmat[0] = bias
    FSList[0] = r"$1$"
    Fmat[1] = TDers[0]
    FSList[1] = r"$u$"
    Fmat[2] = TDers[0] * TDers[0]
    FSList[2] = r"$u^2$"
    Fmat[3] = TDers[1]
    FSList[3] = r"${u,}_{x}$"
    Fmat[4] = TDers[2]
    FSList[4] = r"${u,}_{y}$"
    Fmat[5] = TDers[1] * TDers[1]
    FSList[5] = r"${u^2,}_{x}$"
    Fmat[6] = TDers[2] * TDers[2]
    FSList[6] = r"${u^2,}_{y}$"
    Fmat[7] = TDers[1] * TDers[2]
    FSList[7] = r"${u,}_{y}{u,}_{x}$"
    Fmat[8] = TDers[3] + TDers[5]
    FSList[8] = r"${u,}_{x x} + {u,}_{y y}$"
    Fmat[9] = TDers[4]
    FSList[9] = r"${u,}_{x y}$"
    Fmat = np.row_stack(Fmat)
    return Fmat, FSList


def FeatNormalize(FeatMat):
    """Normalizes the given Feature matrix by dividing every column
    to the element with maximum magnitude. Therefore column values
    scale to [-1, 1].

    Inputs
    ------
    FeatMat : array of doubles with shape = (numfeat, numsamp) where numsamp
    is the number of data samples and numfeat is the number of features.
    It consists the derivatives and products of derivatives etc. for the
    field variable but it is transposed.

    Outputs
    -------
    FeatMat : Column-Normalized version of the input.

    NormVals :array of doubles with shape = (numterms,) where numterms is
    the number of features of original feature matrix. It consists of the
    element with the maximum magnitude for every columns.
    """
    numfeat = len(FeatMat)
    NormVals = np.zeros((numfeat))
    for i in range(numfeat):
        NormVals[i] = np.max(FeatMat[i])
        FeatMat[i] = FeatMat[i] / NormVals[i]
    return FeatMat, NormVals


def STRidge(Fmat, VelVec, lambda1, lambda2, numfeat):
    """STRidge regression algorithm for the sparse regression
    problem. For detailed information go:
    Ref:[Samuel H Rudy, Steven L Brunton, Joshua L Proctor,
    and J Nathan Kutz. Data-driven discovery of partial
    differential equations. Science Advances, 3(4):e1602614, 2017.]

    Inputs
    ------
    Fmat : array of doubles with shape = (numsamp, numfeat) where numsamp is the
    number of data samples and numfeat is the number of features. It consists the
    derivatives and products of derivatives etc. for the field variable.

    VelVec : array of doubles with shape = (numsamp,). It contains the time
    derivative for the field variable.

    Lambda1 : double. Sparsity penalty parameter.

    Lambda2 : double. L2 penalty parameter.

    numfeat : integer. Number of features in matrix Fmat.

    Outputs
    -------
    w : array of doubles with shape = (numfeat,). Contains the coefficients
    of the regression.
    """
    I = np.eye(numfeat)
    w = np.linalg.solve(np.matmul(Fmat.T, Fmat) + lambda2 * I, (Fmat.T).dot(VelVec))
    for k in range(10):
        smallids = np.argwhere(abs(w) < lambda1)
        bigids = np.argwhere(abs(w) >= lambda1).ravel()
        I = np.eye(len(bigids))
        w[smallids] = 0
        w[bigids] = np.linalg.solve(
            np.matmul(Fmat[:, bigids].T, Fmat[:, bigids]) + lambda2 * I,
            (Fmat[:, bigids].T).dot(VelVec),
        )
    return w


dt = 0.05
delta = 4.015
dx = 10.0 / 80
order = 2
numtime = 101
bootindx = np.arange(100)
Tcs = np.load("Coords_xy.npy", allow_pickle=True)

Tvals = np.load("TVals_xy.npy", allow_pickle=True)
Phivals = np.load("PhiVals_xy.npy", allow_pickle=True)

""" 
TDers = [[] for x in range((order + 1) * (order + 2) // 2)]

Fams, Coords, Ts, Vels = FamGen(Tcs, delta, dx, Phivals, Tvals, numtime-1, dt)

for i in range(numtime-1):
    print(i)
    Dmat_T, Derlist = GenDmat(dx, delta, order, Fams[i], Coords[i], 'T')
    for k in range(len(Derlist)):
        TDers[k].append(csr_matrix.dot(Dmat_T[k], Ts[i])) 


np.save('TDers_xy.npy', TDers, allow_pickle=True)
np.save('Derlist_xy.npy', Derlist, allow_pickle=True)

np.save('Vels_xy.npy', Vels, allow_pickle=True)
"""
Vels = np.load("Vels_xy.npy", allow_pickle=True)
TDers = np.load("TDers_xy.npy", allow_pickle=True)
Derlist = np.load("Derlist_xy.npy", allow_pickle=True)
Fmat, FSList = FmatCrt(TDers, Derlist, bootindx)
FeatMat, NormVals = FeatNormalize(Fmat)
VelVec = np.concatenate([Vels[k] for k in bootindx])

velnor = max(VelVec)
print(velnor)
VelVec = VelVec / velnor
w = STRidge(FeatMat.T, VelVec, 0.2, 1.0, len(FeatMat))
err = mean_squared_error(VelVec, np.dot(Fmat.T, w))
w = np.divide(w.ravel(), NormVals) * velnor
print(err)
print(w)
