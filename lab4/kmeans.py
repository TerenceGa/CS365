
# SYSTEM IMPORTS
from typing import Callable, List, Type
import numpy as np


# PYTHON PROJECT IMPORTS



# CONSTANTS
EPSILON: float = 1e-12


# TYPES DEFINED
KMeansType: Type = Type["KMeans"]


def check_2d(X: np.ndarray) -> None:
    if len(X.shape) != 2:
        raise ValueError(f"ERROR: expected argument X to be 2d but got shape {X.shape}")


def random_init(X: np.ndarray,
                num_pts: int
                ) -> np.ndarray:
    """
        A function to return a bunch of random points within the same feature space as the input data X
        @param X: the input data (will use this to lookup the dimensionality of each random point)
        @param num_pts: the number of random points to make
        @returns a matrix where each row is a random point
    """
    check_2d(X)

    _, dim = X.shape
    return np.random.randn(num_pts, dim)


def kmeans_plus_plus_init(X: np.ndarray,
                          num_pts: int
                          ) -> np.ndarray:
    """
        A function to return a bunch of points within the same feature space as the input data X using the kmeans++ initialization algorithm
        @param X: the input data (will use this to lookup the dimensionality of each random point)
        @param num_pts: the number of semi-random points to make
        @returns a matrix where each row is a semi-random point
    """
    check_2d(X)

    num_examples, dim = X.shape

    random_pts: np.ndarray = np.empty((num_pts, dim), dtype=float)

    # choose random point as first cluster center
    random_pts[0, :] = X[np.random.randint(num_examples), :]

    # assign the rest of the points
    for cluster_idx in range(1, num_pts):
        # want to get the euclidean distance from each sample point to the closest cluster
        # We could calculate the euclidean distance from each point to each existing cluster
        # but that would be a massive matrix that won't scale well when we have lots of data
        # so lets do this with a for loop

        min_squared_euclidean_dists: np.ndarray = np.empty((num_examples, 1), dtype=float)
        for pt_idx in range(num_examples):
            clusters_to_get_dist_to: np.ndarray = random_pts[:cluster_idx+1, :]
            pt: np.ndarray = (X[pt_idx, :]).reshape(1, -1) # make sure it is a row vector
            distances: np.ndarray = np.linalg.norm(clusters_to_get_dist_to - pt,
                                                   ord=2,
                                                   axis=1) ** 2
            min_squared_euclidean_dists[pt_idx, 0] = distances.min()

        # now pick a point proportional to those probabilities
        probs: np.ndarray = min_squared_euclidean_dists / min_squared_euclidean_dists.sum()
        random_pts[cluster_idx] = X[np.random.choice(np.arange(num_examples), p=probs.reshape(-1)), :]
    return random_pts


class KMeans(object):
    def __init__(self: KMeansType,
                 num_clusters: int,
                 initialization_type: str = None,
                 init_centers: np.ndarray = None
                 ) -> None:
        self.num_clusters: int = int(num_clusters)
        self.initialization_func: Callable[[np.ndarray, int], np.ndarray] = None

        if initialization_type is None:
            initialization_type = "random"

        if initialization_type.lower() == "random":
            self.initialization_func = random_init
        elif initialization_type.lower() == "kmeans++":
            self.initialization_func = kmeans_plus_plus_init

        # initialize our prior for the gaussians...initially uniform
        # because I don't have any advanced knowledge of which gaussian is more likely than another
        self.cluster_centers: np.ndarray = None
        if init_centers is not None:
            self.cluster_centers = np.copy(init_centers)

    def cost(self: KMeansType,
             X: np.ndarray
             ) -> float:
        cost: float = 0.0

        cluster_idx_assignments: np.ndarray = self.estep(X)
        for pt_idx, cluster_idx in enumerate(cluster_idx_assignments.reshape(-1)):
            cost += np.linalg.norm(self.cluster_centers[cluster_idx] - X[pt_idx],
                                   ord=2)**2 # squared euclidean distance

        return cost

    def check_convergence(self: KMeansType,
                          old_centers: np.ndarray,
                          new_centers: np.ndarray
                          ) -> bool:
        if old_centers.shape != new_centers.shape:
                raise RuntimeError(f"ERROR: self.cluster_centers has changed shape!" +
                                   f"Was {old_centers.shape} and now is {new_centers.shape}")

        return np.all(np.linalg.norm(new_centers - old_centers, ord=2, axis=1) <= EPSILON)

    def estep(self: KMeansType,
              X: np.ndarray
              ) -> np.ndarray:
        check_2d(X)
        num_examples, _ = X.shape

        # cluster_idx_assignments here will contain the cluster index ("idx") that each point was assigned to.
        # In hard em/clustering we assign a point deterministically to a single cluster where that cluster is the "best" fit
        # for the point.

        # all this means is that this variable will be a column vector of cluster indices. Say for example that cluster_idx_assignments[2, 0] = 3
        # that means point 2 in the data (i.e. "X") was assigned to cluster 3 (the fourth cluster).
        # We assign a point to the closest cluster. In KMeans we do this by measuring distance using squared euclidean distance (squared l-2 norm)

        # One note here is that if there is a tie for the closest cluster, settle the tie using the smallest cluster idx
        # (so when considering points 4, if cluster 2, 5, and 6 all tie for the closest cluster to point 4,
        #  choose cluster 2 as it is the smallest cluster idx out of all choices)
        # TODO: finish me!
        cluster_idx_assignments: np.ndarray = np.empty((num_examples, 1), dtype=int)
        dist = np.sum((X[:, np.newaxis, :] - self.cluster_centers)**2, axis=2)
        cluster_idx_assignments = np.argmin(dist, axis=1).reshape(-1, 1)
        return cluster_idx_assignments


    def mstep(self: KMeansType,
          X: np.ndarray,
          cluster_idx_assignments: np.ndarray
          ) -> None:
        check_2d(X)
        num_examples, _ = X.shape

        new_cluster_centers: np.ndarray = np.empty_like(self.cluster_centers)

        for cluster_idx in range(self.num_clusters):
            assigned_pts: np.ndarray = X[cluster_idx_assignments.flatten() == cluster_idx]
            if assigned_pts.size > 0:
                new_center = np.mean(assigned_pts, axis=0)
                new_cluster_centers[cluster_idx] = new_center
            else:
                new_cluster_centers[cluster_idx] = self.cluster_centers[cluster_idx]
        
        # Update self.cluster_centers with the new values
        self.cluster_centers = new_cluster_centers


    def em(self: KMeansType,
           X: np.ndarray
           )-> None:
        cluster_idx_assignments: np.ndarray = self.estep(X)
        self.mstep(X, cluster_idx_assignments)

    def fit(self: KMeansType,
            X: np.ndarray,
            monitor_func: Callable[[KMeansType, int, bool], None] = None, # function to call during each iteration of the fit process
            max_iters: int = int(1e6),      # how many iterations to try before giving up
            delta: float = 1e-9             # convergence threshold for log likelihood between iterations
            ) -> KMeansType:

        # initialize cluster centers
        if self.cluster_centers is None:
            self.cluster_centers = self.initialization_func(X, self.num_clusters)

        current_iter: int = 0
        has_converged: bool = False

        old_cluster_centers: np.ndarray = np.copy(self.cluster_centers) # deepcopy
        while current_iter < max_iters and not has_converged:

            # self.cluster_centers should be updated here
            self.em(X)

            has_converged = self.check_convergence(old_cluster_centers,
                                                   self.cluster_centers)
            np.copyto(old_cluster_centers, self.cluster_centers) # deepcopy with existing buffer
            current_iter += 1
            if monitor_func is not None:
                monitor_func(self, current_iter, has_converged)

        return self



def main() -> None:

    # some helper functions to check that things are working correctly
    def check_centers_shape(centers: np.ndarray,
                            k: int,
                            dim: int
                            ) -> None:
        check_2d(centers)

        # check that number of rows in centers == k
        if centers.shape[0] != k:
            raise ValueError(f"ERROR: expected centers to have {k} rows")
        if centers.shape[-1] != dim:
            raise ValueError(f"ERROR: expected centers to have {dim} cols")


    def check_cluster_idx_assignments(X: np.ndarray,
                                      cluster_idx_assignments: np.ndarray,
                                      k: int
                                      ) -> None:
        check_2d(cluster_idx_assignments)

        # check that we get one cluster idx per pt
        if X.shape[0] != cluster_idx_assignments.shape[0]:
            raise ValueError(f"ERROR: expected cluster_idx_assignments to have same # of rows as X")
        if cluster_idx_assignments.shape[-1] != 1:
            raise ValueError(f"ERROR: cluster_idx_assignments should be a column vector")

        # check (min, max) elements of cluster_idx_assignments are in range
        min_idx_assignment: int = cluster_idx_assignments.min()
        max_idx_assignment: int = cluster_idx_assignments.max()

        if min_idx_assignment < 0:
            raise ValueError("ERROR: assigned cluster idxs  must be >= 0")
        if max_idx_assignment >= k:
            raise ValueError(f"ERROR: assigned cluster idxs must be < k={k}")


    def check_estep(X: np.ndarray,
                    cluster_idx_assignments: np.ndarray,
                    centers: np.ndarray,
                    k: int
                    ) -> None:

        # check the shapes
        check_2d(X)

        check_cluster_idx_assignments(X, cluster_idx_assignments, k)
        check_centers_shape(centers, k, X.shape[-1])

        # check that each assignment was correct
        for pt_idx, assigned_cluster_idx in enumerate(cluster_idx_assignments.reshape(-1)):
            # get closest cluster idx to this pt with our way of handling ties from E-step
            distances_to_each_cluster: np.ndarray = np.linalg.norm(centers - X[pt_idx, :].reshape(1, -1),
                                                                   ord=2,
                                                                   axis=1) ** 2
            closest_cluster_idx: int = np.argmin(distances_to_each_cluster)
            if closest_cluster_idx != assigned_cluster_idx:
                raise ValueError(f"ERROR: closest cluster idx = {closest_cluster_idx} "
                      + f"for point {pt_idx}={X[pt_idx, :]}...distances to each cluster "
                      + "f= {distances_to_each_cluster}")


    def check_mstep(X: np.ndarray,
                    cluster_idx_assignments: np.ndarray,
                    old_centers: np.ndarray,
                    new_centers: np.ndarray,
                    k: int,
                    delta: float = 1e-7
                    ) -> None:
        # check the shapes
        check_2d(X)

        check_cluster_idx_assignments(X, cluster_idx_assignments, k)

        check_centers_shape(old_centers, k, X.shape[-1])
        check_centers_shape(new_centers, k, X.shape[-1])

        for cluster_idx in range(k):
            # get the points assigned to this cluster
            pts_assigned_to_me: np.ndarray = X[cluster_idx_assignments.reshape(-1) == cluster_idx, :]

            new_center: np.ndarray = old_centers[cluster_idx, :]
            if pts_assigned_to_me.shape[0] > 0:
                new_center = pts_assigned_to_me.mean(axis=0)

            distance: float = np.linalg.norm(new_center - new_centers[cluster_idx, :])
            if(distance >= delta):
                raise ValueError(f"ERROR expected new center for cluster {cluster_idx} to be {new_center} "
                                 + f"but was {new_centers[cluster_idx, :]} which is {distance} away!")


    from sklearn.datasets import load_iris

    max_iters: int = 1000
    datasets_loaders = [load_iris]
    dataset_names = ["iris"]

    for init_method in ["random", "kmeans++"]:
        print(f"testing model with {init_method} initialization method")

        # make sure estep is working first
        for dataset_loader, dataset_name in zip(datasets_loaders, dataset_names):
            print(f"\ttesting estep method on the {dataset_name} dataset")
            X: np.ndarray = dataset_loader().data

            for k in [3, 5, 10, 15]:
                print(f"\t\ttesting estep with k={k} clusters")
                m: KMeans = KMeans(k, initialization_type=init_method) # could have clusters which don't get assigned any pts
                m.cluster_centers = m.initialization_func(X, m.num_clusters) # manually call this in this step...normally called by fit()

                for iter_idx in range(max_iters):
                    check_estep(X, m.estep(X), m.cluster_centers, k)

        # make sure mstep is working separately
        for dataset_loader, dataset_name in zip(datasets_loaders, dataset_names):
            print(f"\ttesting mstep method on the {dataset_name} dataset")
            X: np.ndarray = dataset_loader().data

            for k in [3, 5, 10, 15]:
                print(f"\t\ttesting mstep with k={k} clusters")
                m: KMeans = KMeans(k, initialization_type=init_method) # could have clusters which don't get assigned any pts
                m.cluster_centers = m.initialization_func(X, m.num_clusters) # manually call this in this step...normally called by fit()

                old_centers: np.ndarray = np.empty_like(m.cluster_centers)
                for iter_idx in range(max_iters):
                    np.copyto(old_centers, m.cluster_centers) # copy over data before we do anything

                    cluster_idx_assignments: np.ndarray = m.estep(X)
                    m.mstep(X, cluster_idx_assignments)

                    # check that mstep worked
                    check_mstep(X, cluster_idx_assignments, old_centers, m.cluster_centers, k)
    print("congrats! Your model seems to be working!")

if __name__ == "__main__":
    main()

