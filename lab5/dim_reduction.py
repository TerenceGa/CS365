# SYSTEM IMPORTS
from typing import Callable, List, Type, Tuple          # typing info
from tqdm import tqdm                                   # progress bar in python
import numpy as np                                      # linear algebra
import os                                               # manipulating paths


# PYTHON PROJECT IMPORTS



# CONSTANTS
CD: str = os.path.abspath(os.path.dirname(__file__))    # get dir to this file
DATA_DIR: str = os.path.join(CD, "data")                # make path relative to this file
CLASS_0_DIR: str = os.path.join(DATA_DIR, "class_0")
CLASS_1_DIR: str = os.path.join(DATA_DIR, "class_1")


# TYPES DEFINED



def load_data() -> Tuple[np.ndarray, np.ndarray]:
    class_0_data: List[np.ndarray] = list()
    class_1_data: List[np.ndarray] = list()

    # load in all the individual numpy arrays and flatten them
    for list_to_populate, path_to_load in zip([class_0_data, class_1_data],
                                              [CLASS_0_DIR,  CLASS_1_DIR]):
        for npy_file in [x for x in os.listdir(path_to_load)
                         if x.endswith(".npy")]:
            list_to_populate.append(np.load(os.path.join(path_to_load, npy_file)).reshape(-1))

    # make a matrix from each list, where each element of the list is a row
    # don't forget to change this into floats!
    return tuple([np.vstack(data_list).astype(float)
                  for data_list in [class_0_data, class_1_data]])


def check_2d(X: np.ndarray) -> None:
    if len(X.shape) != 2:
        raise ValueError(f"ERROR: expected X to be 2d but had shape {X.shape}!")


def check_same_num_examples(X: np.ndarray,
                            Y: np.ndarray
                            ) -> bool:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"ERROR: expected X & Y to have same # of rows: X={X.shape}, Y={Y.shape}")


def randomly_project(X: np.ndarray,                     # the original dataset
                     k: int                             # the dimensionality to reduce to
                     ) -> Tuple[np.ndarray, np.ndarray]:
    # this function should return a pair:
    #   (f, X_reduced)
    check_2d(X)
    n, d = X.shape
    A = np.random.randn(d, k)
    f = (1 / np.sqrt(k)) * A
    X_reduced = X @ f
    return f, X_reduced


def check_if_distance_satisfied(X: np.ndarray,          # the original dataset
                                X_reduced: np.ndarray,  # the reduced dataset
                                epsilon: float          # how far away points can be without breaking constraints
                                ) -> bool:
    # this function should return False if any of the points in X break a constraint
    # and True otherwise
    check_2d(X)
    check_2d(X_reduced)
    check_same_num_examples(X, X_reduced)

    n = X.shape[0]

    X_squre_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    X_reduced_squre_norm = np.sum(X_reduced ** 2, axis=1).reshape(-1, 1)

    D_orig = X_squre_norm + X_squre_norm.T - 2 * np.dot(X, X.T)
    D_reduced = X_reduced_squre_norm + X_reduced_squre_norm.T - 2 * np.dot(X_reduced, X_reduced.T)

    D_orig = np.maximum(D_orig, 0)
    D_reduced = np.maximum(D_reduced, 0)

    triu_indices = np.triu_indices(n, k=1)
    orig_distances = D_orig[triu_indices]
    reduced_distances = D_reduced[triu_indices]

    lower_bounds = (1 - epsilon) * orig_distances
    upper_bound = (1 + epsilon) * orig_distances

    withn_lower = reduced_distances >= lower_bounds
    within_upper = reduced_distances <= upper_bound

    return np.all(withn_lower & within_upper)

def reduce_dims_randomly(X: np.ndarray, k: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, int]:
    iteration = 0
    max_iter = 10000  # Increased maximum iterations
    while iteration < max_iter:
        iteration += 1
        f, X_reduced = randomly_project(X, k)
        if check_if_distance_satisfied(X, X_reduced, epsilon):
            return f, X_reduced, iteration
    raise RuntimeError(f"No valid projection found within {max_iter} iterations.")

        


def main() -> None:
    X_class_0, X_class_1 = load_data()
    print([X.shape for X in [X_class_0, X_class_1]])

    X: np.ndarray = np.vstack([X_class_0, X_class_1])
    print(X.shape)


    # if we find 1000 random projections, the average number of iterations
    # should converge to at most the expected number of iterations
    # which we can calculate knowing that the number of iterations is a geometric random variable.
    # what is the probability of success?
    num_samples: int = 1000
    iter_samples: List[int] = list()

    for _ in tqdm(range(num_samples)):
        _, _, num_iter = reduce_dims_randomly(X, 32, 0.3)
        iter_samples.append(num_iter)

    print("avg number of iterations=", np.mean(iter_samples))


if __name__ == "__main__":
    main()

