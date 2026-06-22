"""Agglomerative clustering for time series."""

from collections.abc import Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance


class TimeSeriesAgglomerative(BaseClusterer):
    """Agglomerative hierarchical clustering for time series.

    Hierarchical clustering [1]_ organises n time series into a nested
    hierarchy of clusters. Agglomerative hierarchical clustering initially
    treats each time series as an individual cluster. It then repeatedly merges
    the two closest clusters until a complete hierarchy is formed. The hierarchy
    can be cut using either a predefined number of clusters or a distance
    threshold.

    Standard hierarchical clustering commonly relies on Euclidean distance,
    which may perform poorly when time series contain temporal shifts or local
    distortions. This implementation combines agglomerative clustering from
    scikit-learn with pairwise elastic distances provided by aeon. It supports
    single, average, and complete linkage strategies. Ward linkage is currently
    excluded [2]_. The estimator follows the interface and conventions of aeon.



    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` when
        ``distance_threshold`` is not ``None``.
    distance_threshold : float or None, default=None
        Linkage distance threshold used to stop merging clusters. If this is
        not ``None``, ``n_clusters`` must be ``None`` and the full tree is
        computed.
    linkage : {"complete", "average", "single"}, default="single"
        Linkage criterion used to decide which pair of clusters to merge:

        - ``"average"`` uses the average pairwise distance between the two
          clusters.
        - ``"complete"`` uses the maximum pairwise distance between the two
          clusters.
        - ``"single"`` uses the minimum pairwise distance between the two
          clusters.

        Ward linkage is not supported because this estimator works with a
        precomputed time-series distance matrix.
    distance : str or Callable, default="msm"
        Distance used by aeon to compute the pairwise time-series distance
        matrix. Any distance accepted by :func:`aeon.distances.pairwise_distance`
        can be used.
    distance_params : dict or None, default=None
        Additional keyword arguments for the distance function.
    compute_distances : bool, default=False
        Whether to store distances between merged clusters. This is useful for
        inspecting or plotting the hierarchy, but increases memory usage.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_cases,)
        Cluster labels for each time series.
    n_clusters_ : int
        The number of clusters found. If ``distance_threshold`` is ``None``,
        this is equal to ``n_clusters``.
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_connected_components_ : int
        Number of connected components.
    children_ : np.ndarray of shape (n_cases - 1, 2)
        Children of each non-leaf node in the hierarchical tree. Values smaller
        than ``n_cases`` refer to original time series. At merge step ``i``,
        ``children_[i, 0]`` and ``children_[i, 1]`` are merged to form node
        ``n_cases + i``.
    distances_ : np.ndarray of shape (n_cases - 1,)
        Linkage distances corresponding to ``children_``. This attribute is
        available only when ``distance_threshold`` is used or
        ``compute_distances=True``.
    distance_matrix_ : np.ndarray of shape (n_cases, n_cases)
        Pairwise time-series distance matrix used for clustering.

    Examples
    --------
    >>> from aeon.datasets import load_gunpoint
    >>> from utils.HC_Scikit import TimeSeriesAgglomerative
    >>> X, y = load_gunpoint(split="train")
    >>> clusterer = TimeSeriesAgglomerative(
    ...     n_clusters=2,
    ...     distance="euclidean",
    ...     linkage="average",
    ... )
    >>> clusterer.fit(X)
    TimeSeriesAgglomerative(distance='euclidean', linkage='average')
    >>> clusterer.labels_.shape
    (50,)

    References
    ----------
    .. [1] Murtagh, F. and Contreras, P. "Algorithms for hierarchical
        clustering: an overview." Wiley Interdisciplinary Reviews:
        Data Mining and Knowledge Discovery, 2(1), 86-97, 2012.

    .. [2] Murtagh, F. and Legendre, P. "Ward's hierarchical agglomerative
        clustering method: which algorithms implement Ward's criterion?"
        Journal of Classification, 31(3), 274-295, 2014.

    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_clusters: int | None = 2,
        distance_threshold: float | None = None,
        linkage: str = "single",
        distance: str | Callable = "msm",
        distance_params: dict | None = None,
        compute_distances: bool = False,
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.distance = distance
        self.distance_params = distance_params
        self.compute_distances = compute_distances

        super().__init__()

    def _fit(self, X, y=None):
        """Fit agglomerative clustering to X."""
        self._check_params(X)

        self.distance_matrix_ = self._pairwise_distance(X)

        estimator = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
            compute_full_tree=self._compute_full_tree(),
            compute_distances=self.compute_distances,
        )

        estimator.fit(self.distance_matrix_)

        self.labels_ = estimator.labels_
        self.n_clusters_ = estimator.n_clusters_
        self.n_leaves_ = estimator.n_leaves_
        self.n_connected_components_ = estimator.n_connected_components_
        self.children_ = estimator.children_

        if hasattr(estimator, "distances_"):
            self.distances_ = estimator.distances_

        self._estimator = estimator

        return self

    def _predict(self, X):
        """Predict cluster labels for X.

        Agglomerative clustering is transductive, so labels are only defined
        for the data seen during fit.
        """
        raise NotImplementedError(
            "TimeSeriesAgglomerative does not support predict. "
            "Use fit_predict(X) or access labels_ after fit(X)."
        )

    def _pairwise_distance(self, X):
        """Compute pairwise time series distances."""
        distance_params = {} if self.distance_params is None else self.distance_params

        dist_mat = pairwise_distance(
            X,
            method=self.distance,
            **distance_params,
        )

        dist_mat = np.asarray(dist_mat, dtype=float)

        if dist_mat.ndim != 2:
            raise ValueError("The distance matrix must be 2-dimensional.")

        if dist_mat.shape[0] != dist_mat.shape[1]:
            raise ValueError("The distance matrix must be square.")

        if dist_mat.shape[0] != len(X):
            raise ValueError("The distance matrix size must match the number of cases.")

        if not np.all(np.isfinite(dist_mat)):
            raise ValueError("The distance matrix contains NaN or infinite values.")

        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0.0)

        return dist_mat

    def _check_params(self, X):
        """Check estimator parameters."""
        allowed_linkages = ("complete", "average", "single")

        if self.linkage == "ward":
            raise ValueError(
                "Ward linkage is not supported with precomputed distances."
            )

        if self.linkage not in allowed_linkages:
            raise ValueError(
                f"linkage must be one of {allowed_linkages}, "
                f"but found {self.linkage!r}."
            )

        if self.n_clusters is None and self.distance_threshold is None:
            raise ValueError(
                "Exactly one of n_clusters and distance_threshold must be set."
            )

        if self.n_clusters is not None and self.distance_threshold is not None:
            raise ValueError(
                "n_clusters and distance_threshold are mutually exclusive."
            )

        if self.n_clusters is not None:
            if not isinstance(self.n_clusters, int):
                raise TypeError("n_clusters must be an int or None.")

            if self.n_clusters < 1:
                raise ValueError("n_clusters must be greater than 0.")

            if self.n_clusters > len(X):
                raise ValueError(
                    "n_clusters cannot be greater than the number of cases."
                )

        if self.distance_threshold is not None:
            if self.distance_threshold < 0:
                raise ValueError(
                    "distance_threshold must be greater than or equal to 0."
                )

        if self.distance_params is not None and not isinstance(
            self.distance_params, dict
        ):
            raise TypeError("distance_params must be a dict or None.")

        if not isinstance(self.compute_distances, bool):
            raise TypeError("compute_distances must be a bool.")

    def _compute_full_tree(self):
        """Return compute_full_tree value for sklearn."""
        if self.distance_threshold is not None:
            return True

        return "auto"
