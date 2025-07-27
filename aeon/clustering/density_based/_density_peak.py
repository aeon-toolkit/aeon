"""Density clustering for time series data."""

__maintainer__ = []
__all__ = ["DensityPeakClusterer"]

import numpy as np

from aeon.distances import get_distance_function, pairwise_distance
from aeon.utils.validation._dependencies import _check_soft_dependencies


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
        Whether to use a Gaussian kernel for density estimation.
    cutoff_distance : float or str, optional
        Distance cutoff for the Gaussian kernel. If set to "auto", the cutoff is
        automatically selected.
    distance_metric : str, default="euclidean"
        The distance metric to use for clustering. Supported metrics include
        "euclidean", "manhattan", "minkowski", "sqeuclidean" (squared euclidean),
        and other metrics supported by aeon.distances.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    density_threshold : float, optional
        Density threshold for selecting cluster centers. If None, the midpoint of min
        and max density is used.
    distance_threshold : float, optional
        Distance threshold for selecting cluster centers. If None, the midpoint of min
        and max delta is used.
    abnormal : bool, default=False
        If True, points in the halo region (border points) are marked with a label
        of -1.
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
        abnormal: bool = False,
    ):
        self.rho = rho
        self.gauss_cutoff = gauss_cutoff
        self.cutoff_distance = cutoff_distance
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.abnormal = abnormal

    def _build_distance(self):
        """
        Compute the pairwise distance matrix using the resolved distance function.

        Returns
        -------
        distance_matrix : np.ndarray
            A symmetric matrix of pairwise distances.
        """
        # Handle squared euclidean distance for compatibility with original DPCA
        if self.distance_metric == "sqeuclidean":
            distance_matrix = pairwise_distance(self.data, method="squared")
        else:
            dist_func = get_distance_function(self.distance_metric)
            distance_matrix = pairwise_distance(self.data, method=dist_func)
        return distance_matrix

    def _auto_select_dc(self):
        """
        Auto-select cutoff distance (dc).

        The fraction of pairwise distances less than dc
        is within a target range (approximately 0.2% to 1%).
        Intended target range: 0.01 <= nneighs <= 0.002.
        """
        tri_indices = np.triu_indices(self.n, k=1)
        distances = self.distance_matrix[tri_indices]
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        dc = (max_distance + min_distance) / 2

        lower_bound = 0.01  # 0.2
        upper_bound = 0.002  # 1

        while True:
            nneighs = np.sum(distances < dc) / (self.n**2)
            if lower_bound <= nneighs <= upper_bound:
                break
            if nneighs < lower_bound:  # too few neighbors => increase dc
                min_distance = dc
            elif nneighs > upper_bound:  # too many neighbors => decrease dc
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
            The chosen cutoff distance.
        """
        if self.cutoff_distance == "auto":
            return self._auto_select_dc()
        elif self.cutoff_distance is None:
            n = self.data.shape[0]
            self.distances = {}
            for i in range(n):
                for j in range(i + 1, n):
                    self.distances[(i, j)] = self.distance_matrix[i, j]
                    self.distances[(j, i)] = self.distance_matrix[i, j]
            percent = 2.0
            position = int(self.n * (self.n + 1) / 2 * percent / 100)
            sorted_vals = np.sort(list(self.distances.values()))
            dc = sorted_vals[position * 2 + self.n]
            return dc
        else:
            return self.cutoff_distance

    def _fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the density peak clusterer to the training data.

        Parameters
        ----------
        X : array-like
            Time series (or 2D) data to cluster.
        y : array-like, optional
            Labels for the data (unused in clustering).

        Returns
        -------
        self : object
            The fitted clusterer.
        """
        self.data = X
        self.n = X.shape[0]

        self.distance_matrix = self._build_distance()
        self.dc = self.select_dc()

        # Compute local density, excluding self-distance
        self.rho = np.zeros(self.n)
        if self.gauss_cutoff:
            for i in range(self.n):
                self.rho[i] = (
                    np.sum(np.exp(-((self.distance_matrix[i] / self.dc) ** 2))) - 1
                )  # -1 to exclude self (diagonal)
        else:
            # Count neighbors using a hard cutoff, subtracting the self-count
            for i in range(self.n):
                self.rho[i] = np.sum(self.distance_matrix[i] < self.dc) - 1

        # computing delta and nearest higher-density neighbor
        self.delta = np.full(self.n, np.inf)
        self.nneigh = np.zeros(self.n, dtype=int)
        sorted_indices = np.argsort(-self.rho)  # indices in descending order of density
        self.sorted_indices = sorted_indices

        highest_index = sorted_indices[0]
        # For the highest-density point, set delta to the maximum distance in the matrix
        self.delta[highest_index] = np.max(self.distance_matrix)
        self.nneigh[highest_index] = highest_index

        for i in range(1, self.n):
            current_index = sorted_indices[i]
            for j in range(i):
                higher_index = sorted_indices[j]
                d = self.distance_matrix[current_index, higher_index]
                if d < self.delta[current_index]:
                    self.delta[current_index] = d
                    self.nneigh[current_index] = higher_index

        # midpoint rule if thresholds are not provided
        if self.density_threshold is None:
            rho_threshold = 0.5 * (self.rho.min() + self.rho.max())
        else:
            rho_threshold = self.density_threshold

        if self.distance_threshold is None:
            delta_threshold = 0.5 * (self.delta.min() + self.delta.max())
        else:
            delta_threshold = self.distance_threshold

        # Initial cluster assignment:
        # marking points as centers,they exceed both density and delta thresholds
        self.labels_ = -np.ones(self.n, dtype=int)
        for idx in range(self.n):
            if (self.rho[idx] >= rho_threshold) and (
                self.delta[idx] >= delta_threshold
            ):
                self.labels_[idx] = idx

        # labels for non-center points based on the nearest higher-density neighbor.
        for idx in sorted_indices:
            if self.labels_[idx] == -1:
                self.labels_[idx] = self.labels_[self.nneigh[idx]]

        # Create a copy for halo assignment.
        halo = self.labels_.copy()
        # Identify unique cluster centers.
        unique_centers = [i for i in range(self.n) if self.labels_[i] == i]
        # Initialize border density for each cluster.
        bord_rho = {center: 0.0 for center in unique_centers}

        # For every pair of points in clusters within dc, update border density
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (
                    self.labels_[i] != self.labels_[j]
                    and self.distance_matrix[i, j] <= self.dc
                ):
                    rho_avg = (self.rho[i] + self.rho[j]) / 2.0
                    if (
                        self.labels_[i] in bord_rho
                        and rho_avg > bord_rho[self.labels_[i]]
                    ):
                        bord_rho[self.labels_[i]] = rho_avg
                    if (
                        self.labels_[j] in bord_rho
                        and rho_avg > bord_rho[self.labels_[j]]
                    ):
                        bord_rho[self.labels_[j]] = rho_avg

        # points falling in halo region (density lower than cluster's border density)
        for i in range(self.n):
            if self.labels_[i] in bord_rho and self.rho[i] < bord_rho[self.labels_[i]]:
                halo[i] = 0

        # if abnormal flag is True, assign halo points a label of -1
        if self.abnormal:
            for i in range(self.n):
                if halo[i] == 0:
                    self.labels_[i] = -1

        # Final cluster centers: points whose label equals their own index.
        self.cluster_centers = [i for i in range(self.n) if self.labels_[i] == i]

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the clusterer to the training data.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_timepoints) or
            (n_samples, n_channels, n_timepoints)
            Data to cluster.
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
        Plot the clustering results and/or the decision graph.

        Parameters
        ----------
        mode : str, default="all"
            One of "decision" (to plot the decision graph),
            "label" (to plot cluster labels), or "all" (to plot both).
        title : str, optional
            Title for the plots.
        kwargs : dict
            Additional keyword arguments passed to plotting functions.
        """
        _check_soft_dependencies("matplotlib")
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
