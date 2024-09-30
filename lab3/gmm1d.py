# SYSTEM IMPORTS
from typing import Callable, List, Type
import numpy as np


# PYTHON PROJECT IMPORTS



# CONSTANTS
EPSILON: float = 1e-12
DELTA: float = 1e-9


# TYPES DEFINED
GMM1DType: Type = Type["GMM1D"]



# This function returns the probability of observing your data given N(mu, variance)
# i.e. how likely it is that N(mu, cov) generated the observed data
def pdf(X: np.ndarray, mu: float, variance: float) -> np.ndarray:
    return np.exp((-((X-mu)**2))/(2*variance)) / np.sqrt(2*np.pi*variance)


class GMM1D(object):
    def __init__(self: GMM1DType,
                 num_gaussians: int) -> None:
        self.num_gaussians: int = num_gaussians

        # initialize our prior for the gaussians...initially uniform
        # because I don't have any advanced knowledge of which gaussian is more likely than another
        self.priors: np.ndarray = np.ones((self.num_gaussians, 1), dtype=float) / self.num_gaussians # col vec

        # start off with random means for our k gaussians
        self.mus: np.ndarray = np.random.randn(self.num_gaussians, 1)   # col vec

        # start off with random variances for our k gaussians. np.random.rand() returns values in [0,1)
        self.variances: np.ndarray = np.random.rand(self.num_gaussians, 1) + EPSILON # make it not 0 by adding EPSILON

    def log_likelihood(self: GMM1DType,
                       X: np.ndarray
                       ) -> float:
        likelihoods: np.ndarray = np.hstack([(prior*pdf(X, mu, variance)).reshape(-1,1) # make column vector
                                             for prior, mu,variance in zip(self.priors.reshape(-1),
                                                                           self.mus.reshape(-1),
                                                                           self.variances.reshape(-1))
                                             ])
        return np.sum(np.log(np.sum(likelihoods, axis=1)))

    def estep(self: GMM1DType,
              X: np.ndarray
              ) -> np.ndarray:
        if len(X.shape) != 2:
            raise ValueError("ERROR: X must have shape (num_examples, 1)")

        num_examples, _ = X.shape

        # gammas here will contain our posterior values. In the lecture slides this was written as
        # Pr[Z_ij | x_i]. Each row in gammas will correspond to an example, while each column corresponds to a cluster
        # (i.e. a Gaussian that that example could have come from)
        gammas: np.ndarray = np.empty((num_examples, self.num_gaussians), dtype=float)

        # gammas will look like:
        #             cluster_1                   cluster_2           ...         cluster_k
        #     x_1  [[Pr[cluster_1 | x_1],   Pr[cluster_2 | x_1],      ...       Pr[cluster_k | x_1]],
        #     x_2   [Pr[cluster_1 | x_2],   Pr[cluster_2 | x_2],      ...       Pr[cluster_k | x_2]],
        #     x_3   [Pr[cluster_1 | x_3],   Pr[cluster_2 | x_3],      ...       Pr[cluster_k | x_3]],
        #     ...            ...                    ...               ...
        #     x_n   [Pr[cluster_1 | x_n],   Pr[cluster_2 | x_n],      ...       Pr[cluster_k | x_n]]


        # Each row here is a pmf and we calculate these values using bayes rule. This means that we can put the
        # numerator values in gamma, and then normalize every row in gamma to end up with the correct probability values

        # TODO: finish me!
        k = self.num_gaussians
        for cluster_idx, (prior, mu, variance) in enumerate(zip(self.priors.reshape(-1),
                                                            self.mus.reshape(-1),
                                                            self.variances.reshape(-1))):
            gammas[:, cluster_idx] = prior * pdf(X, mu, variance).reshape(-1)
        
        denominator = gammas.sum(axis=1, keepdims=True) + EPSILON  # Shape: (num_examples, 1)
        gammas /= denominator

        return gammas


    def mstep(self: GMM1DType,
              X: np.ndarray,
              gammas: np.ndarray
              ) -> None:

        if len(X.shape) != 2:
            raise ValueError("ERROR: X must have shape (num_examples, 1)")

    
    num_examples, _ = X.shape
    k = self.num_gaussians

    # Sum of responsibilities for each Gaussian component (sum_i gamma_ij)
    gammas_sum = gammas.sum(axis=0, keepdims=True).T  # Shape: (k, 1)

    # Update priors: pi_j = sum_i gamma_ij / n
    self.priors = gammas_sum / num_examples  # Shape: (k, 1)

    # Update means: mu_j = sum_i gamma_ij * x_i / sum_i gamma_ij
    # Compute weighted sum of X for each component
    weighted_sum = np.dot(gammas.T, X)  # Shape: (k, 1)
    self.mus = weighted_sum / (gammas_sum + EPSILON)  # Shape: (k, 1)

    # Update variances: sigma_j^2 = sum_i gamma_ij * (x_i - mu_j)^2 / sum_i gamma_ij
    # Compute squared differences between X and mu_j for each component
    # Expand mus to match X's shape for broadcasting
    mus_expanded = self.mus.T  # Shape: (1, k)
    X_expanded = X  # Shape: (num_examples, 1)
    squared_diff = (X_expanded - mus_expanded) ** 2  # Shape: (num_examples, k)

    # Compute weighted sum of squared differences
    weighted_sum_sq = np.dot(gammas.T, squared_diff)  # Shape: (k, 1)
    self.variances = weighted_sum_sq / (gammas_sum + EPSILON)  # Shape: (k, 1)

    # Ensure variances are not too small
    self.variances += EPSILON


    def em(self: GMM1DType,
           X: np.ndarray
           )-> None:
        gammas: np.ndarray = self.estep(X)
        self.mstep(X, gammas)

    def fit(self: GMM1DType,
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

    for k in range(2, 10):

        real_mus: np.ndarray = np.random.randn(k)
        real_vars: np.ndarray = np.random.rand(k) + EPSILON # variances cant be 0 so add EPSILON

        X: np.ndarray = np.vstack([np.random.normal(loc=rmu, scale=rvar, size=num_samples).reshape(-1,1)
                                   for rmu, rvar in zip(real_mus, real_vars)])


        for max_iters in [100, 1000, 10000, 1000000]:

            # if correctly implemented, the log-likelihood should monotonically increase (or plateau)
            m: GMM1D = GMM1D(k)
            # print("init ll: %s" % m.log_likelihood(X))
            lls: List[float] = m.fit(X, max_iters=max_iters)

            if len(lls) == 0:
                raise RuntimeError("1d test FAILED. No log-likelihoods were recorded")

            # convert lls into np array
            lls = np.array(lls, dtype=float)
            if (lls[:-1] - lls[1:]).max() > DELTA:
                raise RuntimeError("1d test FAILED. Log-likelihood did not monotonically increase")
        print(f"1d test PASSED with {k}-clusters")


if __name__ == "__main__":
    main()


