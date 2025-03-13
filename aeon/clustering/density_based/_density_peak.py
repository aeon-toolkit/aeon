"""Density clustering for time series data."""

__maintainer__ = []
__all__ = ["DensityPeakClusterer"]

import numpy as np

from aeon.distances import get_distance_function, pairwise_distance


class DensityPeakClusterer:
    """
    Density Peak Clusterer.

    Clusters time series data using a density-based approach that estimates local
    densities and identifies peaks as cluster centers.

    Parameters
    ----------
    rho : float, optional
        The local density for each data point.
    delta : np.ndarray
        For each point, the minimum distance to any point with higher density.
    gauss_cutoff : bool, default=True
        Whether to use Gaussian cutoff for density estimation.
    cutoff_distance : float, optional
        Distance cutoff for Gaussian kernel.
    distance_metric : str, default="euclidean"
        Distance metric to use for clustering.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    density_threshold : float, optional
        Density threshold to select cluster centers. If None, will use midpoint.
    distance_threshold : float, optional
        Distance threshold to select cluster centers. If None, will use midpoint.
    """

    def __init__(
        self,
        rho: float = None,
        gauss_cutoff: bool = True,
        cutoff_distance: int = None,
        distance_metric: str = "euclidean",
        n_jobs: int = 1,
        density_threshold: float = None,
        distance_threshold: float = None,
    ):
        self.rho = rho
        self.gauss_cutoff = gauss_cutoff
        self.cutoff_distance = cutoff_distance
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold

    def _build_distance(self):
        """
        Compute the pairwise distance matrix using the resolved distance function.

        Returns
        -------
        distance_matrix : np.ndarray
            A matrix of symmetric pairwise distances.
        """
        dist_func = get_distance_function(self.distance_metric)
        distance_matrix = pairwise_distance(self.data, method=dist_func)
        return distance_matrix

    def _auto_select_dc(self):
        """Auto-select cutoff distance (dc) so that the fraction of pairwise distances less than dc is within a target range (e.g: 1-2%)."""  # noqa
        tri_indices = np.triu_indices(
            self.n, k=1
        )  # k=1 to ignore diagonal as it has 0's
        distances = self.distance_matrix[tri_indices]
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        dc = (max_distance + min_distance) / 2

        lower_bound = 0.002
        upper_bound = 0.01  # 1-2% range

        # recursively setting the value of dc
        while True:
            nneighs = np.sum(distances < dc) / (self.n**2)  # nearest neighbors

            if lower_bound < nneighs < upper_bound:  # case 1 : dc is in range
                break
            if nneighs < lower_bound:  # case 2 : increase dc (too few neighbors)
                min_distance = dc
            if nneighs > upper_bound:  # case 3 : decrease dc (too many neighbors)
                max_distance = dc

            dc = (max_distance + min_distance) / 2
            if max_distance - min_distance < 1e-6:
                break

        return dc

    def select_dc(self):
        """
        Select the cutoff distance (dc) for density estimation.

        Returns
        -------
        dc : float
            The cutoff distance.
        """
        if self.cutoff_distance == "auto":
            return self._auto_select_dc()
        else:
            return self.cutoff_distance

    def _fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : array-like
            Time series data to cluster.
        y : array-like, optional
            Labels for the data (unused in clustering).

        Returns
        -------
        self : object
            The fitted clusterer.
        """
        self.data = X
        self.n = X.shape[0]  # total no. of time points

        self.distance_matrix = self._build_distance()  # pairwise distance matrix

        self.dc = self.select_dc()  # selecting cutoff distance

        self.rho = np.zeros(self.n)  # local density

        if self.gauss_cutoff:
            # Gaussian kernel: weight = exp(- (d / self.dc) ** 2)
            for i in range(self.n):
                self.rho[i] = np.sum(
                    np.exp(-((self.distance_matrix[i] / self.dc) ** 2))
                )
        else:
            # Hard cutoff: weight = 1 if d < dc, 0 otherwise
            for i in range(self.n):
                self.rho[i] = np.sum(self.distance_matrix[i] < self.dc)

        self.delta = np.full(self.n, np.inf)
        self.nneigh = np.zeros(self.n, dtype=int)
        sorted_indices = np.argsort(
            -self.rho
        )  # decending order of local density indices
        self.sorted_indices = sorted_indices

        highest_index = sorted_indices[0]
        self.delta[highest_index] = np.max(
            self.distance_matrix
        )  # distance to highest density point
        self.nneigh[highest_index] = highest_index

        for i in range(1, self.n):
            current_index = sorted_indices[i]
            for j in range(i):
                higher_index = sorted_indices[j]
                d = self.distance_matrix[current_index, higher_index]
                if d < self.delta[current_index]:
                    self.delta[current_index] = d
                    self.nneigh[current_index] = higher_index

        # If thresholds are not provided, use the midpoint rule
        if self.density_threshold is None:
            rho_threshold = 0.5 * (self.rho.min() + self.rho.max())
        else:
            rho_threshold = self.density_threshold

        if self.distance_threshold is None:
            delta_threshold = 0.5 * (self.delta.min() + self.delta.max())
        else:
            delta_threshold = self.distance_threshold

        # Initialize cluster labels to -1 and assign centers based on the thresholds
        self.labels_ = -np.ones(self.n, dtype=int)
        for idx in range(self.n):
            if (self.rho[idx] >= rho_threshold) and (
                self.delta[idx] >= delta_threshold
            ):
                self.labels_[idx] = (
                    idx  # mark as cluster center (label equals its own index)
                )

        # descending-density assignment for non-center points
        for idx in sorted_indices:
            if self.labels_[idx] == -1:
                self.labels_[idx] = self.labels_[self.nneigh[idx]]

        # store the final center indices for reference.
        self.cluster_centers = [i for i in range(self.n) if self.labels_[i] == i]

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit time series clusterer to training data.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_timepoints) or
            (n_samples, n_channels, n_timepoints)
            Time series data to cluster.
        y : array-like, optional
            Labels for the data (unused in clustering).

        Returns
        -------
        self : DensityPeakClusterer
            The fitted clusterer.
        """
        self._fit(X)
        self.is_fitted = True
        return self

    def plot(self, mode="all", title="", **kwargs):
        """
        Plot clustering results and/or the decision graph.

        Parameters
        ----------
        mode : str, default="all"
            One of "decision" (plot decision graph), "label" (plot clustered data),
            or "all" (plot both).
        title : str, optional
            Title for the plots.
        kwargs : dict
            Additional keyword arguments passed to plotting functions.
        """
        import matplotlib.pyplot as plt

        if mode in {"decision", "all"}:
            plt.figure()
            plt.scatter(self.rho, self.delta, c=self.labels_, cmap="viridis")
            plt.xlabel("Local Density (rho)")
            plt.ylabel("Delta")
            plt.title(title + " Decision Graph")
            plt.colorbar()
            plt.show()

        if mode in {"label", "all"}:
            plt.figure()
            if self.data.ndim == 2 and self.data.shape[1] >= 2:
                plt.scatter(
                    self.data[:, 0], self.data[:, 1], c=self.labels_, cmap="viridis"
                )
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
            else:
                plt.plot(self.data.T)
            plt.title(title + " Cluster Labels")
            plt.colorbar()
            plt.show()
