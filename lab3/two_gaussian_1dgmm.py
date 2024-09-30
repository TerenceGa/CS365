# SYSTEM IMPORTS
from typing import Callable, List, Type
import matplotlib.pyplot as plt
import numpy as np


# PYTHON PROJECT IMPORTS



# CONSTANTS
EPSILON: float = 1e-12
DELTA: float = 1e-9


# TYPES DEFINED
TwoGaussian1DGMMType: Type = Type["TwoGaussian1DGMM"]



# This function returns the probability of observing your data given N(mu, variance)
# i.e. how likely it is that N(mu, cov) generated the observed data
def pdf(X: np.ndarray, mu: float, variance: float) -> np.ndarray:
    return np.exp((-((X-mu)**2))/(2*variance)) / np.sqrt(2*np.pi*variance)


class TwoGaussian1DGMM(object):
    def __init__(self: TwoGaussian1DGMMType) -> None:
        self.num_gaussians: int = 2

        # initialize our prior for the gaussians...initially uniform
        # because I don't have any advanced knowledge of which gaussian is more likely than another
        self.prior_cluster_1: float = float(1/2)
        self.prior_cluster_2: float = float(1/2)

        # start off with random means for our two gaussians
        self.mu_cluster_1: float = np.random.randn()
        self.mu_cluster_2: float = np.random.randn()

        # start off with random variances for our two gaussians. np.random.rand() returns values in [0,1)
        self.variance_cluster_1: float = np.random.rand() + EPSILON # make it not 0
        self.variance_cluster_2: float = np.random.rand() + EPSILON # make it not 0

    def log_likelihood(self: TwoGaussian1DGMMType,
                       X: np.ndarray
                       ) -> float:
        likelihoods: np.ndarray = np.hstack([pdf(X, mu, variance).reshape(-1,1) # make column vector
                                             for mu,variance in zip([self.mu_cluster_1, self.mu_cluster_2],
                                                                    [self.variance_cluster_1, self.variance_cluster_2])
                                             ])
        likelihoods *= np.array([[self.prior_cluster_1, self.prior_cluster_2]])
        return np.sum(np.log(np.sum(likelihoods, axis=1)))

    def estep(self: TwoGaussian1DGMMType,
              X: np.ndarray
              ) -> np.ndarray:
        if len(X.shape) != 2:
            raise ValueError("ERROR: X must have shape (num_examples, 1)")

        num_examples, _ = X.shape

        # gammas here will contain our posterior values. In the lecture slides this was written as
        # Pr[Z_ij | x_i]. Each row in gammas will correspond to an example, while each column corresponds to a cluster
        # (i.e. a Gaussian that that example could have come from)
        gammas: np.ndarray = np.empty((num_examples, 2), dtype=float)

        # gammas will look like:
        #             cluster_1                   cluster_2
        #     x_1  [[Pr[cluster_1 | x_1],   Pr[cluster_2 | x_1]],
        #     x_2   [Pr[cluster_1 | x_2],   Pr[cluster_2 | x_2]],
        #     x_3   [Pr[cluster_1 | x_3],   Pr[cluster_2 | x_3]],
        #     ...            ...                    ...
        #     x_n   [Pr[cluster_1 | x_n],   Pr[cluster_2 | x_n]]]


        # Each row here is a pmf and we calculate these values using bayes rule. This means that we can put the
        # numerator values in gamma, and then normalize every row in gamma to end up with the correct probability values

        # to populate the numerators, I am going to do this one column (i.e. cluster) at a time
        for cluster_idx, (prior, mu, variance) in enumerate(zip([self.prior_cluster_1, self.prior_cluster_2],
                                                                [self.mu_cluster_1, self.mu_cluster_2],
                                                                [self.variance_cluster_1, self.variance_cluster_2])):
            gammas[:, cluster_idx] = prior * pdf(X, mu, variance).reshape(-1) # array must be 1d to assign into a col


        # now normalize each row by the sum of that row. The sum of a row is the denominator in bayes rule
        # where we use the law of total probability to break it down into cases (and add up the likelihood
        # of each case)
        gammas /= (gammas.sum(axis=1, keepdims=True) + EPSILON) # keepdims will make the output be a column vector here

        return gammas


    def mstep(self: TwoGaussian1DGMMType,
              X: np.ndarray,
              gammas: np.ndarray
              ) -> None:

        if len(X.shape) != 2:
            raise ValueError("ERROR: X must have shape (num_examples, 1)")

        num_examples, _ = X.shape

        # this is where we will apply our MLE formulas:
        #   the MLE of the prior is the mean of the gamma values for that cluster
        self.prior_cluster_1 = np.mean(gammas[:, 0])
        self.prior_cluster_2 = np.mean(gammas[:, 1])

        # just to make sure that these sum to 1 (b/c of machine precision errors) let's also normalize them
        prior_sum: float = self.prior_cluster_1 + self.prior_cluster_2
        self.prior_cluster_1 /= prior_sum
        self.prior_cluster_2 /= prior_sum

        # now we can update each cluster mean
        # the MLE for a gaussian mean (for a cluster) is a weighted sum of the examples
        # where the weight of an example is the gamma value for that example divided by the gammas for that cluster
        # so we can compute the weights first (as a column vector)
        # do some element-wise addition and then add up all of the terms
        # we add EPSILON to the denominator to make sure we arent dividing by zero (b/c of machine precision errors)
        weights_col_vec_cluster_1: np.ndarray = (gammas[:, 0] / (gammas[:, 0].sum() + EPSILON)).reshape(-1, 1)
        weights_col_vec_cluster_2: np.ndarray = (gammas[:, 1] / (gammas[:, 1].sum() + EPSILON)).reshape(-1, 1)

        self.mu_cluster_1 = (weights_col_vec_cluster_1 * X).sum()
        self.mu_cluster_2 = (weights_col_vec_cluster_2 * X).sum()


        # now we can update each cluster variance
        # the MLE for a gaussian variance (for a cluster) is the weighted sum of the example variances
        # so we need to calculate the variance of each sample using our new updated means and then do the weighted
        # combo like before
        X_variances_cluster_1: np.ndarray = (X - self.mu_cluster_1)**2
        X_variances_cluster_2: np.ndarray = (X - self.mu_cluster_2)**2

        self.variance_cluster_1 = (weights_col_vec_cluster_1 * X_variances_cluster_1).sum()
        self.variance_cluster_2 = (weights_col_vec_cluster_2 * X_variances_cluster_2).sum()


    def em(self: TwoGaussian1DGMMType,
           X: np.ndarray
           )-> None:
        gammas: np.ndarray = self.estep(X)
        self.mstep(X, gammas)

    def fit(self: TwoGaussian1DGMMType,
            X: np.ndarray,
            max_iters: int = int(1e6),      # how many iterations to try before giving up
            delta: float = 1e-9             # convergence threshold for log likelihood between iterations
            ) -> List[float]:
        log_likelihoods: List[float] = list()

        current_iter: int = 0
        prev_ll: float = np.inf
        current_ll: float = 0.0

        while current_iter < max_iters and abs(prev_ll - current_ll) > delta:
            self.em(X)

            prev_ll = current_ll
            current_ll = self.log_likelihood(X)
            log_likelihoods.append(current_ll)
            current_iter += 1

        return log_likelihoods



def main() -> None:
    print("running 1d test")
    num_samples: int = 100

    real_mus: np.ndarray = np.array([-4, 4], dtype=float)
    real_vars: np.ndarray = np.array([1.2, 0.8], dtype=float)

    X: np.ndarray = np.vstack([np.random.normal(loc=rmu, scale=rvar, size=num_samples).reshape(-1,1)
                               for rmu, rvar in zip(real_mus, real_vars)])


    for max_iters in [100, 1000, 10000, 1000000]:

        # if correctly implemented, the log-likelihood should monotonically increase (or plateau)
        m: TwoGaussian1DGMM = TwoGaussian1DGMM()
        print("init ll: %s" % m.log_likelihood(X))
        lls: List[float] = m.fit(X, max_iters=max_iters)

        if len(lls) == 0:
            raise RuntimeError("1d test FAILED. No log-likelihoods were recorded")

        # convert lls into np array
        lls = np.array(lls, dtype=float)

        # print(np.abs((lls[:-1] - lls[1:]).max()))

        if (lls[:-1] - lls[1:]).max() > DELTA:
            raise RuntimeError("1d test FAILED. Log-likelihood did not monotonically increase")
    print("1d test PASSED")


if __name__ == "__main__":
    main()


