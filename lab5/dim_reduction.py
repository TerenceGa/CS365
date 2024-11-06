# SYSTEM IMPORTS
from typing import Callable, List, Type, Tuple          # typing info
from tqdm import tqdm                                   # progress bar in python
import numpy as np                                      # linear algebra
import os                                               # manipulating paths
from scipy.spatial.distance import pdist, squareform

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


def check_if_distance_satisfied(X: np.ndarray,          # Original dataset
                                X_reduced: np.ndarray,  # Reduced dataset
                                epsilon: float          # Allowed deviation in distances
                                ) -> bool:
    """
    Checks whether the pairwise distances between points in the original dataset X
    and the reduced dataset X_reduced satisfy the Johnson-Lindenstrauss (JL) constraints.

    Parameters:
    - X (np.ndarray): Original data matrix of shape (n_samples, n_features).
    - X_reduced (np.ndarray): Reduced data matrix of shape (n_samples, n_reduced_features).
    - epsilon (float): Maximum allowed relative distortion in distances.

    Returns:
    - bool: True if all pairwise distances satisfy the JL constraints, False otherwise.
    """

    # Ensure that input matrices are 2D and have the same number of samples
    check_2d(X)
    check_2d(X_reduced)
    check_same_num_examples(X, X_reduced)

    n_samples = X.shape[0]

    # Compute pairwise Euclidean distances in the original and reduced spaces
    D_X = squareform(pdist(X, metric='euclidean'))
    D_X_reduced = squareform(pdist(X_reduced, metric='euclidean'))

    # Create a mask to avoid division by zero (when original distances are very small)
    nonzero_mask = D_X > 1e-8

    # Initialize the ratios array with zeros
    ratios = np.zeros_like(D_X)

    # Compute the ratio of distances where original distances are non-zero
    ratios[nonzero_mask] = D_X_reduced[nonzero_mask] / D_X[nonzero_mask]

    # Define the acceptable range based on epsilon
    lower_bound = 1 - epsilon
    upper_bound = 1 + epsilon

    # Identify violations where the ratio is outside the acceptable range
    violations = (ratios < lower_bound) | (ratios > upper_bound)

    # Ignore self-distances by setting the diagonal to False
    np.fill_diagonal(violations, False)

    # If any violations are found, return False; otherwise, return True
    return not np.any(violations)

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

