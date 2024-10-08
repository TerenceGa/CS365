# SYSTEM IMPORTS
from abc import abstractmethod, ABC     # you need these to make an abstract class in Python
from typing import List, Type, Union    # Python's typing syntax
import numpy as np
import math                      # linear algebra & useful containers
from math import comb


# PYTHON PROJECT IMPORTS


# Types defined in this module
DistributionType: Type = Type["Distribution"]
BinomialDistributionType = Type["BinomialDistribution"]
PoissonDistributionType = Type["PoissonDistribution"]
GaussianDistributionType = Type["GaussianDistribution"]

# an abstract class for an arbitrary distribution
# please don't touch this class
class Distribution(ABC):

    # this is how you make an abstract method in Python
    # all child classes MUST implement this method
    # otherwise we will get an error when the child class
    # is instantiated 
    @abstractmethod 
    def fit(self: DistributionType,
            X: np.ndarray               # input data to fit from
            ) -> DistributionType:
        ... # same as "pass"

    # another abstract method that every child class will have to implement
    # in order to be able to be instantiated
    @abstractmethod
    def prob(self: DistributionType,
             X: np.ndarray
             ) -> np.ndarray:           # return Pr[x] for each point (row) of the data (X)
        ... # same as "pass"

    @abstractmethod
    def parameters(self: DistributionType) -> List[Union[float, np.ndarray]]:
        ... # same as "pass"


# a class for the binomial distribution
# you will need to complete this class
class BinomialDistribution(Distribution):
    def __init__(self: BinomialDistributionType,
                 n: int) -> None:
        # controlled by parameter "p"
        self.p: float = None
        self.n: int = n

    def fit(self: BinomialDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> BinomialDistributionType:
        if X.shape[-1] != self.n:
            raise ValueError
        if X.shape[0] != 1:
            raise ValueError(f"ERROR: expected a single row to fit from")

        totalsuscesses = np.sum(X)
        self.p = totalsuscesses / self.n



        return self # keep this at the end


    def prob(self: BinomialDistributionType, X: np.ndarray) -> np.ndarray:
        if X.shape[-1] != self.n:
            raise ValueError(f"ERROR: expected 2D data with {self.n} outcomes in each example")
        k = np.sum(X, axis=1)
        P = np.power(self.p, k) * np.power(1 - self.p, self.n - k)
        return P.reshape(-1, 1)
    
    def parameters(self: BinomialDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.n, self.p]



class GaussianDistribution(Distribution):
    def __init__(self: GaussianDistributionType) -> None:
        # controlled by parameters mu and var
        self.mu: float = None
        self.var: float = None

    def fit(self: GaussianDistributionType,
            X: np.ndarray                   # input data to fit from
                                            # this will be a bunch of integer samples stored in a column vector
            ) -> GaussianDistributionType:

        self.mu = np.mean(X)
        self.var = np.var(X, ddof=0)
        return self # keep this at the end

    def prob(self: GaussianDistributionType,
             X: np.ndarray                  # this will be a column vector where every element is a float
             ) -> np.ndarray:
        c = 1.0 / np.sqrt(2 * np.pi * self.var)
        exponent = -((X - self.mu) ** 2) / (2 * self.var)
        p = c * np.exp(exponent)
        return p.reshape(-1, 1)

    def parameters(self: GaussianDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.mu, self.var]



# a class for the poisson distribution
# you will need to complete this class
class PoissonDistribution(Distribution):
    def __init__(self: PoissonDistributionType) -> None:
        # controlled by parameter "lambda"
        self._lambda: float = None

    def fit(self: PoissonDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> PoissonDistributionType:
        X = X.flatten()
        self._lambda = np.mean(X)
        return self
    

    def prob(self: PoissonDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        X = X.flatten()
        k = X.astype(np.float64)
        lambda_ = self._lambda

        # Compute log PMF to improve numerical stability
        log_pmf = -lambda_ + k * math.log(lambda_) - np.array([math.lgamma(val + 1) for val in k])
        probabilities = np.exp(log_pmf)
        
        return probabilities.reshape(-1, 1)

    def parameters(self: PoissonDistributionType) -> List[Union[float, np.ndarray]]:
        return [self._lambda]

