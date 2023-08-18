import os
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from matplotlib.ticker import FormatStrFormatter
from math import comb as comb_n
import operator

# Define the operators for the creation of the library terms
ops = {"+": operator.add, "*": operator.mul}
opsltx = {"+": "+", "*": "\\times "}
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
# Font size and type settings for the plots
params = {
    "text.usetex": True,
    "font.size": 10,
    "font.family": "lmodern",
}
plt.rcParams.update(params)


class BGEL:
    def __init__(self, u_ders, vels, lambda_2, numtime, model_name):
        self.u_ders = u_ders
        self.vels = vels
        self.lambda_2 = lambda_2
        self.numtime = numtime
        self.boot_data_idx = []
        self.coeffs = None
        self.can_lib = None
        self.lib_terms = None
        self.inc_threshold = None
        self.nboot_data = None
        self.nboot_lib = None
        self.numfeat = None
        self.term_comb = None
        self.operators = None
        self.model_name = model_name
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

    def bootstrap_data_lib(self, nboot_data: int, nboot_lib: int, numfeat: int):
        """
        Bootstraps the datapoints and library terms for given amounts. Where numfeat
        is the number of features in the feature matrix. This number includes the column
        of ones at the first position
        """
        self.nboot_data = nboot_data
        self.nboot_lib = nboot_lib
        self.numfeat = numfeat
        self.can_lib = [[] for _ in range(numfeat)]
        self.coeffs = [[] for _ in range(nboot_data)]
        combvec = np.arange(0, numfeat)
        # Bootstrap libraries without replacement
        self.lib_terms = list(itertools.combinations(combvec, nboot_lib))
        # Bootstrap datapoints with replacement
        for _ in range(nboot_data):
            self.boot_data_idx.append(
                np.sort(
                    np.random.choice(range(self.numtime), self.numtime, replace=True)
                )
            )

    def crt_can_lib(
        self, term_comb: list[list[int]], operators: list[str], der_list: list[str]
    ):
        """
        Creates library columns for given term and operation combinations
        """
        self.term_comb = term_comb
        self.operators = operators
        # Adding 1 as a candidate term
        self.can_lib[0] = r"$1$"
        for i in range(self.numfeat - 1):
            operator = self.operators[i]
            term_str = r"$"
            # Combining opeartors and the terms by replacing operators
            # in between terms
            for j in term_comb[i][:-1]:
                term_str += der_list[j]
                term_str += opsltx[operator[0]]
            term_str += der_list[term_comb[i][-1]]
            term_str += "$"
            self.can_lib[i + 1] = term_str

    def calc_inc_prob(self, inc_threshold: float, disp_thresholded: bool = True):
        """
        Calculates the inclusion probabilities using the resulting coefficient appearences in different
        Bootstrapped datasets. Takes the threshold probability as the input
        """
        # Scaling the probability using the amount of experiments each library term is left out
        scale_prob = comb_n(self.numfeat, self.nboot_lib) / comb_n(
            self.numfeat - 1, self.nboot_lib - 1
        )
        self.inc_threshold = inc_threshold
        # Empty vector for storing number of occurences
        num_oc = np.zeros(self.numfeat)
        for i in range(self.nboot_data):
            for j in range(len(self.lib_terms)):
                # Collect all the instances that a coefficient appears as nonzero
                num_oc += np.where(self.coeffs[i][j] == 0, self.coeffs[i][j], 1)
        # Calculate the inclusion probabilities by dividing the number of occurences
        # times the scaling factor to total number of experiments
        self.incprob = scale_prob * num_oc / (self.nboot_data * len(self.lib_terms))
        # Eliminate all the terms appearing with less than threshold probability
        if not disp_thresholded:
            self.incprob = np.where(self.incprob > self.inc_threshold, self.incprob, 0)

    def calc_med_std(self, print_median: bool = True, discard_tiny: bool = False):
        """
        This calculates the medians, means and standard deviations of the coefficients
        """
        self.med_coeffs = np.zeros(self.numfeat)
        self.std_coeffs = np.zeros(self.numfeat)
        self.mean_coeffs = np.zeros(self.numfeat)
        for i in range(self.numfeat):
            valvec = []
            for j in range(self.nboot_data):
                for k in range(len(self.lib_terms)):
                    if i in self.lib_terms[k]:
                        valvec.append(self.coeffs[j][k][i])
            valvec = [val for val in valvec if val != 0]
            if self.incprob[i] > 0:
                self.med_coeffs[i] = np.median(valvec)
                self.std_coeffs[i] = np.std(valvec)
                self.mean_coeffs[i] = np.mean(valvec)
        if print_median:
            print("Recovered Coefficients")
            res = "\n".join(
                "{} {:.3f} w.p.-> {:.2f}".format(x, y, z)
                for x, y, z in zip(self.can_lib, self.med_coeffs, self.incprob)
            )
            print(res)
        # Disregard terms appearing with really small coeffs, which basically means the
        # sparsity coefficients is not large enough for some subsets of library bagging
        if discard_tiny:
            self.incprob = np.where(abs(self.med_coeffs) > 1e-3, self.incprob, 0)

    def plt_coeff_dist(self):
        """
        Plots the probability distributions for recovered coefficients
        """
        self.coeffs = [item for sublist in self.coeffs for item in sublist]
        nnzvec = np.nonzero(self.incprob)[0]
        nnzlen = len(nnzvec)
        numexp = len(self.coeffs)
        libname = []
        cfplt = []
        cfplti = np.zeros(numexp)
        for i in range(nnzlen):
            for k in range(numexp):
                cfplti[k] = self.coeffs[k][nnzvec[i]]
            cfplt.append(cfplti[cfplti != 0.0])
            libname.append(self.can_lib[nnzvec[i]])
        for j in range(nnzlen):
            title_color = (
                "green" if self.incprob[nnzvec[j]] > self.inc_threshold else "red"
            )
            fig = plt.figure(figsize=(1.5, 1.5))
            ax = fig.gca()
            # Kernel density estimation of the parameter using the discovered
            # values of different experiments
            pdist = gaussian_kde(cfplt[j])
            # Borders of the plots are chosen as 3*sigma far from the mean
            # of the coefficient
            xleft = np.mean(cfplt[j]) - 3 * np.std(cfplt[j])
            xright = np.mean(cfplt[j]) + 3 * np.std(cfplt[j])
            xmedian = self.med_coeffs[nnzvec[j]]
            # Marking the border and median values
            xticks = (
                [xleft, xmedian, xright]
                if self.incprob[nnzvec[j]] > self.inc_threshold
                else [xmedian]
            )
            # Creating a grid for plot
            card_space = np.linspace(xleft, xright, 1000)
            ax.plot(card_space, pdist(card_space))
            # Marking the median with a vertical line
            ax.vlines(x=xmedian, ymin=0, ymax=pdist(xmedian))
            # Removing the ticks for y
            ax.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )
            plt.xticks(xticks)
            ax.ticklabel_format(axis="x", style="sci", scilimits=[-2, 2])
            # Filling below the PDFs with uniform red color
            ax.fill_between(
                card_space, pdist(card_space), color="red", alpha=0.1, linewidth=0
            )
            plt.ylabel(libname[j], math_fontfamily="cm")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            plt.title(
                f"$P_{{inc}}=$" + f"{self.incprob[nnzvec[j]]: .3f}",
                math_fontfamily="cm",
                color=title_color,
            )
            plt.tight_layout()
            plt.savefig(self.model_name + "//coeff_uncertainty" + str(j), dpi=1200)
            plt.close()

    def bagg_learn(self):
        """
        This applies STRidge to bootstrapped datasets
        """
        for i in range(self.nboot_data):
            feat_mat = BGEL.crt_feat_mat(
                self.u_ders,
                self.boot_data_idx[i],
                self.term_comb,
                self.operators,
                self.numfeat,
            )
            # Stacking the velocity vector values
            vel_vec = np.concatenate([self.vels[k] for k in self.boot_data_idx[i]])
            # Normalization of the Feature matrix
            feat_mat, vel_vec, norm_vals = BGEL.normalize_system(
                feat_mat, vel_vec, self.numfeat
            )
            # Sparsity parameter is fixed to 0.3, however it can be changed if the
            # random ensemble doesn't return any value with P>P_threshold
            lambda_1 = 0.3
            for j, can_terms in enumerate(self.lib_terms):
                self.coeffs[i].append(np.zeros(len(feat_mat)))
                feat_mat_red = feat_mat[can_terms, :]
                w = BGEL.st_ridge(
                    feat_mat_red.T, vel_vec, lambda_1, self.lambda_2, len(feat_mat_red)
                )
                # Reverting the normalization and placing the recovered coefficients to
                # a list
                np.put(
                    self.coeffs[i][j],
                    can_terms,
                    (
                        np.divide(w.ravel(), np.take(norm_vals[:-1], can_terms).ravel())
                        * norm_vals[-1]
                    ),
                )

    @staticmethod
    def crt_feat_mat(
        u_ders: np.ndarray,
        boot_idx: np.ndarray,
        term_comb: list[list[int]],
        operators: list[list[str]],
        numfeat: int,
    ):
        """
        Creates the Feature matrix using the array of the field variable and the derivatives
        up to specified order, bootstrapped datapoint indices, desired term combinations,
        the operators for the combinations and the number of features
        """
        feat_mat = [[] for _ in range(numfeat)]
        for i in range(numfeat - 1):
            operator = operators[i]
            feat_mat[i + 1] = np.concatenate(
                u_ders[term_comb[i][0]][boot_idx[:]]
            ).ravel()
            for j in term_comb[i][1:]:
                feat_mat[i + 1] = ops[operator[0]](
                    feat_mat[i + 1], np.concatenate(u_ders[j][boot_idx[:]]).ravel()
                )
        feat_mat[0] = np.ones_like(feat_mat[1])
        return np.row_stack(feat_mat)

    @staticmethod
    def normalize_system(fmat: np.ndarray, velvec: np.ndarray, numfeat: int):
        """
        Normalizes the given Feature matrix by dividing every column
        to the element with maximum magnitude. Therefore column values
        scale to [-1, 1]
        """
        normvals = np.ones((numfeat + 1))
        for i in range(numfeat):
            normvals[i] = max(abs(fmat[i]))
            fmat[i] = fmat[i] / normvals[i]
        normvals[-1] = 1.0
        return fmat, velvec, normvals

    @staticmethod
    def st_ridge(fmat: np.ndarray, velvec: np.ndarray, lambda_1: float, lambda_2: float, num_feat: int):
        """
        STRidge regression algorithm for the sparse regression
        problem. For detailed information go:
        Ref:[Samuel H Rudy, Steven L Brunton, Joshua L Proctor,
        and J Nathan Kutz. Data-driven discovery of partial
        differential equations. Science Advances, 3(4):e1602614, 2017.]
        """
        I = np.eye(num_feat)
        w = np.linalg.solve(
            np.matmul(fmat.T, fmat) + lambda_2 * I, (fmat.T).dot(velvec)
        )
        for _ in range(10):
            smallids = np.argwhere(abs(w) < lambda_1)
            bigids = np.argwhere(abs(w) >= lambda_1).ravel()
            I = np.eye(len(bigids))
            w[smallids] = 0
            w[bigids] = np.linalg.solve(
                np.matmul(fmat[:, bigids].T, fmat[:, bigids]) + lambda_2 * I,
                (fmat[:, bigids].T).dot(velvec),
            )
        nz_idx = np.where(w != 0)[0]
        w[nz_idx] = np.linalg.lstsq(fmat[:, nz_idx], velvec, rcond=-1)[0]
        return w

    @staticmethod
    def calcbic(fmat: np.ndarray, velvec: np.ndarray, lambda_2: float, lambda_1f: float, lambda_1c: float, eps: float, numl1samp: int):
        """
        Calculates optimum sparsity parameter using bayesian information
        criterion for given data and sparsity parameter interval. An extra
        error of eps is added to prevent over-fitting
        """
        lambda_1vec = np.logspace(np.log10(lambda_1f), np.log10(lambda_1c), numl1samp)
        bic_score = np.zeros(numl1samp)
        numdata = len(fmat[0])
        for i in range(numl1samp):
            w = BGEL.st_ridge(fmat.T, velvec, lambda_1vec[i], lambda_2, len(fmat))
            nnz = np.count_nonzero(w)
            err = mean_squared_error(velvec, np.dot(fmat.T, w))
            bic_score[i] = np.log(err + eps) * numdata + nnz * np.log(numdata)
        plot1fig(lambda_1vec, bic_score, r"$\lambda$", "BIC Score", "OptLambd")
        minidx = np.argmin(bic_score)
        lambda_opt = lambda_1vec[minidx]
        return lambda_opt


def plot1fig(x_data, y_data, xname, yname, figname):
    fig = plt.figure(1, figsize=(7, 7), dpi=1200)
    plt.plot(
        x_data,
        y_data,
        marker="o",
        markersize=10,
        mfc="white",
        linestyle="--",
        alpha=1.0,
    )
    plt.xlabel(xname, fontsize=14, math_fontfamily="cm")
    plt.ylabel(yname, fontsize=14, math_fontfamily="cm")
    plt.savefig(figname)
