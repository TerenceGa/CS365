from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


####################################################################################################
## MODEL PARAMETERS
# this is a class containing a numpy array of parameters
# the purpose of this class is to keep the gradient of a parameter and the value of that parameter
# in the same class.
class Parameter(object):
    def __init__(self, X: np.ndarray) -> None:
        self.val: np.ndarray = X
        self.grad: np.ndarray = None

    def reset(self) -> "Parameter":
        self.grad = np.zeros_like(self.val)
        return self

    def step(self, G: np.ndarray) -> "Parameter":
        self.val -= G
        return self.reset()
####################################################################################################


####################################################################################################
## THE MODULE BASE CLASS
# A "module" is a building block in the block-diagram of a model.
# Each module needs to know how to process input (i.e. the "forward" method),
# how to compute its tiny part of chain rule and combine it with the overall derivative
# so far (i.e. the "backwards" method),
# and an interface to it's parameters (if this module has any)
class Module(ABC):

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """ This method defines how the module "processes" input data.

            @param X. This is a numpy array (in batch form) of inputs.

            @out: Y_hat. This output value of this module (in batch form).
        """
        ...

    @abstractmethod
    def backward(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        """ This method defines two things:
                1) the tiny piece of chain rule this module is associated with.
                2) how it combines it's piece of chain rule with the overall derivative
                   so far

            @param: X. This is a numpy array (in batch form) of inputs just like inputs
                    to self.forward
            @param: dLoss_dModule. This is a numpy array which caches the total derivative
                    from the loss function backwards to the output of this module.
                    This numpy array has the same shape as self.forward(X).

            @out: dLoss_dX. This is the total derivative from the loss function
                  (in batch form) backwards to the output of the *previous* module
                  (whose output is input to this module). This numpy array should have
                  the same shape as X.
        """
        ...

    @abstractmethod
    def parameters(self) -> List[Parameter]:
        """ This method returns a list of Parameter objects. This list may be empty
            if this module does not have any learnable parameters.

            @out: List[Parameter]. The list of Parameter objects this module has (allowed
                  to be empty).
        """
        ...
####################################################################################################


####################################################################################################
## LOSS FUNCTIONS
# The base class for a loss function. Technically you can consider this a Module
# however it has a slightly different API since it is the start of the derivative
# so it gets it's own class heirarchy.
class LossFunction(ABC):

    @abstractmethod
    def forward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> float:
        """ This method produces a total error from the two batch form numpy arrays.

            @param Y_hat. This numpy array (in batch form) is the predictions of the model
            @param Y_gt. This numpy array (in batch form) is the ground truth

            @out float. The expected error (lower is better)
        """
        ...

    @abstractmethod
    def backward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> np.ndarray:
        """ This method is the start of the derivative for gradient descent. This method
            should compute the tiny part of chain rule that the loss function is responsible
            for (i.e. dLoss/dY_hat).

            @param Y_hat. This numpy array (in batch form) is the predictions of the model
            @param Y_gt. This numpy array (in batch form) is the ground truth

            @out dLoss/dY_hat. This is the total derivative of the expected loss with
                 respect to Y_hat. This should have the same shape as Y_hat.
        """
        ...


# Binary Cross Entropy loss function, which we derived in class. Given predictions y_hat_0, y_hat_1, ... y_hat_N
# and the correct ground truth y_gt_0, y_gt_1, ... y_gt_N, the cross entropy loss is measured as:
#
#       L(y_hats, y_gts) = (1/N) * sum_{i=1 -> N}(y_gt_i*log(y_hat_i) + (1-y_gt_i)*log(1-y_hat_i))
#
# the (1/N) coefficient we didn't talk about in class, but it helps to keep our gradients numerically stable
class BCE(LossFunction):

    def check_input(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> None:
        # Y_hat and Y_gt must have same shape
        assert(Y_hat.shape == Y_gt.shape)

        # Y_gt must contain Pr[c_0] so must be a column vector
        assert(len(Y_gt.shape) == 2 and Y_gt.shape[-1] == 1)

    # compute the loss
    def forward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> float:
        # formula for a single example: (-1/m) * (y_gt*log(y_hat) + (1-y_gt)*log(1-y_hat))

        self.check_input(Y_hat, Y_gt)

        return (np.sum(-np.log(Y_hat[Y_gt != 0])) +\
                np.sum(-np.log(1-Y_hat[Y_gt == 0]))) / Y_hat.shape[0]

    # compute the derivative of the loss with respect to each y_hat_i
    # This method should return a numpy array of the same shape as Y_hat
    # each entry in this numpy array should be the derivative of the loss with respect to a particular y_hat_i
    def backward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> np.ndarray:
        self.check_input(Y_hat, Y_gt)

        # Avoid division by zero
        dLoss_dY_hat_pre = - (Y_gt / Y_hat - (1 - Y_gt) / (1 - Y_hat)) 


        N = Y_hat.shape[0]
        X = 1 / dLoss_dY_hat_pre
        X /= N
        dLoss_dY_hat = X

        
        # your dLoss_dY_hat should have the same shape as Y_hat
        assert(dLoss_dY_hat.shape == Y_hat.shape)
        return dLoss_dY_hat
####################################################################################################



####################################################################################################
## OPTIMIZERS (how the gradient and the parameters are combined together)
# this class is the base class for how we take the "step" in gradient descent.
class Optimizer(ABC):
    def __init__(self, parameters: List[Parameter], lr: float) -> None:
        self.parameters: List[Parameter] = parameters
        self.lr: float = lr

    def reset(self) -> None:
        for p in self.parameters:
            p.reset()

    @abstractmethod
    def step(self) -> None:
        ...


# so far we've only learned how to take the vanilla gradient descent step
class GradientDescentOptimizer(Optimizer):

    def step(self) -> None:
        for p in self.parameters:
            p.step(self.lr * p.grad)
####################################################################################################


# Sigmoid activation function
class Sigmoid(Module):

    # calculates sigmoid(x) element wise. So if x is multi-dimensional, we apply sigmoid to each element
    def forward(self, X: np.ndarray) -> np.ndarray:
        # the formula of sigmoid is 1/(1 + e^-x)
        return 1 / (1 + np.exp(-X))


    # calculates the derivative of the loss with respect to the input of sigmoid
    # this is broken up into two parts thanks to chain rule
    # we know that the output of sigmoid is used to calculate somehow the predictions, and the predictions
    # are combined in some loss function
    #
    # so if we know dLoss/dSigmoid_output (called dLoss_dModule),
    # then we can calculate dLoss/dX as dSigmoid_output/dX * dLoss/dSigmoid_output
    def backward(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        sigmoid_X = self.forward(X)
        # TODO: compute dModule_dX (tiny piece of chain rule)
        dModule_dX: np.ndarray = np.zeros_like(X)
        dModule_dX = sigmoid_X * (1 - sigmoid_X)
        # TODO: combine dModule_dX and dLoss_dModule to get dLoss_dX
        dLoss_dX: np.ndarray = np.zeros_like(X)
        dLoss_dX = dLoss_dModule * dModule_dX
        # your dLoss_dX should have the same shape as X
        assert(dLoss_dX.shape == X.shape)
        return dLoss_dX

    def parameters(self) -> List[Parameter]:
        # sigmoid has no parameters
        return list()


class LinearRegression(Module):
    def __init__(self, num_features: int) -> None:
        self.betas: np.ndarray = Parameter(np.random.randn(num_features+1, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        # I don't like concatenating 1s because it make this take longer than it should
        return np.dot(X, self.betas.val[:-1, :]) + self.betas.val[-1:,:]

    def backward(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:

        # TODO: compute dModule_dBeta, combine it with dLoss_dModule to get dLoss_dBeta
        dModule_dBeta: np.ndarray = np.zeros_like(self.betas.val)
        dLoss_dBeta: np.ndarray = np.zeros_like(self.betas.val)
        dLoss_dBeta = np.vstack([
            np.dot(X.T, dLoss_dModule),                # Gradient for beta coefficients
            np.sum(dLoss_dModule, axis=0, keepdims=True)  # Gradient for intercept beta0
        ]) 
        # please don't change these lines.
        # your dLoss_dBeta should be the same shape as self.betas.val
        assert(dLoss_dBeta.shape == self.betas.val.shape)
        self.betas.grad += dLoss_dBeta

        # TODO: compute dModule_dX (tiny piece of chain rule)
        dModule_dX: np.ndarray = np.zeros_like(X)
        dModule_dX: np.ndarray = self.betas.val[:-1, :]
        # TODO: combine dModule_dX and dLoss_dModule to get dLoss_dX
        dLoss_dX: np.ndarray = np.zeros_like(X)
        dLoss_dX: np.ndarray = np.dot(dLoss_dModule, dModule_dX.T)
        # your dLoss_dX should have the same shape as X
        assert(dLoss_dX.shape == X.shape)
        return dLoss_dX

    def parameters(self) -> List[Parameter]:
        return [self.betas]


class LogisticRegression(Module):
    def __init__(self, num_features: int) -> None:
        self.num_features: int = num_features
        self.linear_regression: Module = LinearRegression(self.num_features)
        self.sigmoid: Module = Sigmoid()

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid.forward(self.linear_regression.forward(X))

    def backward(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        # compute loss through sigmoid module
        A: np.ndarray = self.linear_regression.forward(X)
        dLoss_dA: np.ndarray = self.sigmoid.backward(A, dLoss_dModule)

        # compute loss through linear regression module
        dLoss_dX: np.ndarray = self.linear_regression.backward(X, dLoss_dA)

        return dLoss_dX

    def parameters(self) -> List[Parameter]:
        return self.linear_regression.parameters() + self.sigmoid.parameters()


# numerical gradient checking...this is how we check whether
# our gradients are correct. What we do is compute the
# numerical partial derivative with respect to each learnable
# parameter. A numerical derivative can be computed using:
#       df/dx = (f(x+e) - f(x-e))/(2*e)
# if we set "e" to be really small, then we can get a good approx
# of the gradient. We can compare the symbolic gradients
# versus the numerical gradients, and hope they are super close
def grad_check(X: np.ndarray, Y_gt: np.ndarray, m: Module, ef: LossFunction,
               epsilon: float = 1e-3, delta: float = 1e-3) -> None:
    params: List[Parameter] = m.parameters()
    num_grads: List[np.ndarray] = [np.zeros_like(P.val) for P in params]
    sym_grads: List[np.ndarray] = [P.grad for P in params]

    for P, N in zip(params, num_grads):
        for index, v in np.ndenumerate(P.val):
            P.val[index] += epsilon
            N[index] += ef.forward(m.forward(X), Y_gt)

            P.val[index] -= 2*epsilon
            N[index] -= ef.forward(m.forward(X), Y_gt)

            # set param back to normal
            P.val[index] = v
            N[index] /= (2*epsilon)

    ratios: np.ndarray = np.array([np.linalg.norm(SG-NG)/
                                   np.linalg.norm(SG+NG)
                                   for SG, NG in zip(sym_grads, num_grads)], dtype=float)
    if np.sum(ratios > delta) > 0:
        raise RuntimeError("ERROR: failed grad check. delta: [%s], ratios: %s"
            % (delta, ratios))


def main() -> None:
    np.random.seed(12345)
    num_features: int = 100
    num_examples: int =  200
    lr: float = 0.1
    max_epochs: int = 1000

    X: np.ndarray = np.random.randn(num_examples, num_features)
    Y_gt: np.ndarray = np.random.choice([0,1], size=(num_examples, 1))

    m: Module = LogisticRegression(num_features)
    optimizer: Optimizer = GradientDescentOptimizer(m.parameters(), lr)
    loss_func: LossFunction = BCE()

    losses: List[float] = list()
    for _ in tqdm(list(range(max_epochs)), desc="training & checking gradients"):
        optimizer.reset()
        Y_hat: np.ndarray = m.forward(X)
        losses.append(loss_func.forward(Y_hat, Y_gt))
        m.backward(X, loss_func.backward(Y_hat, Y_gt))
        grad_check(X, Y_gt, m, loss_func)
        optimizer.step()

    msg = "INFO: If you see this message (AND you have grad_check enabled " +\
          " AND if MAX_EPOCHS > 0) then your code is WORKING so CONGRATS!!"
    print(msg)
    plt.plot(losses)
    plt.xlabel("training iteration")
    plt.ylabel("$\mathbb{E}[L(\hat{y}, y_{gt})]$")
    plt.show()


if __name__ == "__main__":
    main()

