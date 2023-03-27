import numpy as np
from math import factorial


class PD2D:
    def __init__(self, delta, dx, numtime):
        self.delta = delta
        self.dx = dx
        self.numtime = numtime

    @staticmethod
    def binomial_coeff(n):
        """
        Calculates binomial coefficients and powers for
        given order.

        Inputs
        ------
        n : integer. Order of the binomial expansion.

        Outputs
        -------
        xlen : integer. Number of terms for the expansion.

        coeffs : list of integers of binomial coefficients.
        shape = [xlen].

        pows1 : list of integer powers of the 1st variable in the
        binomial expansion.  shape = [xlen].

        pows2 : list of integer powers of the 2nd variable in the
        binomial expansion.  shape = [xlen].
        """
        xlen = n * (n + 1) // 2
        coeffs = []
        pows1 = []
        pows2 = []
        for i in range(n):
            coeffs += [int(x) for x in str(11**i)]
            pows1 += list(reversed(range(i + 1)))
            pows2 += list(range(i + 1))
        return coeffs, pows1, pows2, xlen

    @staticmethod
    def gen_bmat(order):
        """
        Calculates b Matrix for G function generation.

        Inputs
        ------
        order : integer. Order of Taylor series expansion for PDDO.

        Outputs
        -------
        bmat : array of integers containing the b matrix values.
        shape = ((order + 1) * (order + 2) // 2,(order + 1) * (order + 2) // 2)
        """
        _, pows1, pows2, _ = PD2D.binomial_coeff(order + 1)
        bdiag = [factorial(n1) * factorial(n2) for n1, n2 in zip(pows1, pows2)]
        bmat = np.diag(bdiag)
        return bmat

    @staticmethod
    def gen_amat(xi, delta, order):
        '''Calculates contribution of a family member with relative
        coordinate xi to A matrix for G function generation.

        Inputs
        ------
        xi : array of doubles with shape = (2,), it is the relative coordinate
        of family member when the point of interest is the origin.

        delta : double. Horizon radius.

        order : integer. Order of Taylor series expansion for PDDO.

        Outputs
        -------
        Amat : array of doubles containing the A matrix values.
        shape = ((order + 1) * (order + 2) // 2,(order + 1) * (order + 2) // 2)
        '''
        ximag = np.linalg.norm(xi)
        _, pow1, pow2, _ = PD2D.binomial_coeff(order + 1)
        Avec = np.power(xi[0], pow1) * np.power(xi[1], pow2)
        w = np.exp(-4.0 * (ximag / delta) ** 2)
        Amat = np.outer(Avec, Avec) * w
        return Amat

    @staticmethod
    def inv_gfunc(xi, delta, amat, order):
        '''
        Calculates calues of peridynamic functions at a given
        material point coordinate inside the family.

        Inputs
        ------
        xi : array of doubles with shape = (2,), it is the relative coordinate
        of family member when the point of interest is the origin.

        delta : double. Horizon radius.

        amat : array with doubles containing a matrix values which are the coefficients
        for the peridynamic G functions.
        shape = ((order + 1) * (order + 2) // 2,(order + 1) * (order + 2) // 2).

        order : integer. Order of Taylor series expansion for PDDO.

        Outputs
        -------
        GFS : array of doubles containing G function values for different order
        derivative calculations for the given material point.
        shape = ((order + 1) * (order + 2) // 2,) and dV is excluded
        from the returned value.
        '''
        ximag = np.linalg.norm(xi)
        _, pow1, pow2, _ = PD2D.binomial_coeff(order + 1)
        Avec = np.power(xi[0], pow1) * np.power(xi[1], pow2)
        w = np.exp(-4.0 * (ximag / delta) ** 2)
        GFS = w * np.matmul(amat.T, Avec)
        return GFS
