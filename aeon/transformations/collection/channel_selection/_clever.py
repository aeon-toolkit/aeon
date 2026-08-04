"""CLeVer channel selection."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["CLeVerCluster", "CLeVerHybrid", "CLeVerRank"]

import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from aeon.transformations.collection.channel_selection.base import BaseChannelSelector


def _stable_descending_order(values):
    """Return descending value order, breaking exact ties by lower index."""
    indices = np.arange(len(values))
    return np.lexsort((indices, -values))


def _case_correlation_eigensystem(X_case, case_index, variance_threshold):
    """Compute one case's ordered correlation eigensystem and component count."""
    centered = X_case - np.mean(X_case, axis=1, keepdims=True)
    channel_norms = np.linalg.norm(centered, axis=1)

    # A scale-aware tolerance avoids dividing by roundoff-level variation while
    # retaining genuinely varying channels even when their overall scale is small.
    channel_scale = np.maximum(1.0, np.max(np.abs(X_case), axis=1))
    zero_tolerance = np.finfo(float).eps * np.sqrt(X_case.shape[1]) * channel_scale
    valid = channel_norms > zero_tolerance
    if not np.any(valid):
        raise ValueError(
            f"All channels in case {case_index} are constant or numerically "
            "constant; CLeVer cannot compute explained variance for this case."
        )

    correlation = np.zeros((X_case.shape[0], X_case.shape[0]), dtype=float)
    standardized = centered[valid] / channel_norms[valid, None]
    valid_correlation = standardized @ standardized.T
    valid_indices = np.flatnonzero(valid)
    correlation[np.ix_(valid_indices, valid_indices)] = valid_correlation
    correlation = (correlation + correlation.T) / 2.0
    correlation[valid_indices, valid_indices] = 1.0

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(-eigenvalues, kind="mergesort")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    negative_tolerance = (
        10
        * np.finfo(float).eps
        * correlation.shape[0]
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if np.any(eigenvalues < -negative_tolerance):
        raise ValueError(
            f"The correlation matrix for case {case_index} is not positive "
            "semidefinite within numerical tolerance."
        )
    # Only negative eigenvalues within the roundoff tolerance reach this clipping.
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)

    total_variance = float(np.sum(eigenvalues))
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError(
            f"Case {case_index} has zero total correlation eigenvalue sum; "
            "CLeVer cannot determine its principal components."
        )
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    cumulative_variance[-1] = 1.0
    n_components = int(
        np.searchsorted(cumulative_variance, variance_threshold, side="left") + 1
    )
    return eigenvectors, n_components


class _BaseCLeVer(BaseChannelSelector):
    """Shared CLeVer correlation, PCA, and DCPC implementation."""

    _tags = {
        "capability:multivariate": True,
        "requires_y": False,
        "X_inner_type": "numpy3D",
    }

    def __init__(self, n_channels=5, variance_threshold=0.8):
        self.n_channels = n_channels
        self.variance_threshold = variance_threshold
        super().__init__()

    def _fit(self, X, y=None):
        """Compute descriptive common principal components and select channels."""
        self._validate_parameters(X.shape[1])
        X = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError("CLeVer requires X to contain only finite values.")

        case_eigenvectors = []
        case_n_components = np.empty(X.shape[0], dtype=int)
        for case_index, X_case in enumerate(X):
            eigenvectors, n_components = _case_correlation_eigensystem(
                X_case, case_index, self.variance_threshold
            )
            case_eigenvectors.append(eigenvectors)
            case_n_components[case_index] = n_components

        self.case_n_components_ = case_n_components
        self.n_components_ = int(np.max(case_n_components))
        self.common_matrix_ = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        for eigenvectors in case_eigenvectors:
            # eigh returns eigenvectors as columns. CLeVer represents the leading
            # principal components as rows, hence the transpose before L.T @ L.
            loadings = eigenvectors[:, : self.n_components_].T
            self.common_matrix_ += loadings.T @ loadings
        self.common_matrix_ = (self.common_matrix_ + self.common_matrix_.T) / 2.0

        common_eigenvalues, common_eigenvectors = np.linalg.eigh(self.common_matrix_)
        common_order = np.argsort(-common_eigenvalues, kind="mergesort")
        # DCPCs are rows and original channels are columns, matching the paper.
        self.dcpc_ = common_eigenvectors[:, common_order[: self.n_components_]].T
        self._select_channels()
        return self

    def _validate_parameters(self, n_input_channels):
        """Validate common CLeVer parameters."""
        if isinstance(self.n_channels, bool) or not isinstance(
            self.n_channels, Integral
        ):
            raise ValueError("n_channels must be an integer.")
        if not 1 <= self.n_channels <= n_input_channels:
            raise ValueError(
                "n_channels must be in the range [1, number of input channels]."
            )
        if isinstance(self.variance_threshold, bool) or not isinstance(
            self.variance_threshold, Real
        ):
            raise ValueError("variance_threshold must be a number in (0, 1].")
        if not np.isfinite(self.variance_threshold) or not (
            0 < self.variance_threshold <= 1
        ):
            raise ValueError("variance_threshold must be a number in (0, 1].")

    def _select_channels(self):
        raise NotImplementedError("Subclasses must implement channel selection.")

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return parameters for aeon estimator checks."""
        return {"n_channels": 1}


class _BaseCLeVerClustering(_BaseCLeVer):
    """Shared k-means stage for CLeVer-Cluster and CLeVer-Hybrid."""

    def __init__(
        self,
        n_channels=5,
        variance_threshold=0.8,
        n_init=20,
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        super().__init__(n_channels=n_channels, variance_threshold=variance_threshold)

    def _validate_parameters(self, n_input_channels):
        super()._validate_parameters(n_input_channels)
        if isinstance(self.n_init, bool) or not isinstance(self.n_init, Integral):
            raise ValueError("n_init must be a positive integer.")
        if self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral):
            raise ValueError("max_iter must be a positive integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if isinstance(self.tol, bool) or not isinstance(self.tol, Real):
            raise ValueError("tol must be a nonnegative finite number.")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a nonnegative finite number.")

    def _cluster_channels(self):
        self.channel_loadings_ = self.dcpc_.T
        # The paper states an unsquared sum of distances. sklearn's conventional
        # k-means instead minimizes squared Euclidean distance (inertia), which is
        # the objective used and exposed by this implementation.
        kmeans = KMeans(
            n_clusters=self.n_channels,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            kmeans.fit(self.channel_loadings_)

        self.cluster_labels_ = kmeans.labels_.copy()
        self.cluster_centers_ = kmeans.cluster_centers_.copy()
        self.inertia_ = float(kmeans.inertia_)
        if len(np.unique(self.cluster_labels_)) != self.n_channels:
            raise ValueError(
                "CLeVer k-means produced fewer non-empty clusters than "
                "n_channels, likely because there are too few distinct channel "
                "loading vectors."
            )


class CLeVerRank(_BaseCLeVer):
    """Select channels by their contributions to the CLeVer DCPCs.

    CLeVer-Rank computes a descriptive common principal-component (DCPC)
    representation jointly from all multivariate time-series cases. It scores each
    original channel by the Euclidean norm of its DCPC loading vector and retains the
    ``n_channels`` highest-scoring channels.

    Parameters
    ----------
    n_channels : int, default=5
        Exact number of original channels to retain.
    variance_threshold : float, default=0.8
        Minimum cumulative explained-variance ratio used to determine the number of
        leading principal components for each case. Must be in ``(0, 1]``.

    Attributes
    ----------
    channels_selected_ : list[int]
        Selected channel indices in their original input order.
    channel_scores_ : np.ndarray of shape (n_input_channels,)
        Euclidean norm of each channel's DCPC loading vector.
    channel_ranking_ : np.ndarray of shape (n_input_channels,)
        All channel indices in descending score order, with lower indices first for
        exact ties.
    dcpc_ : np.ndarray of shape (n_components_, n_input_channels)
        Descriptive common principal components stored as rows.
    n_components_ : int
        Maximum number of components required by any individual case.
    case_n_components_ : np.ndarray of shape (n_cases,)
        Number of components required to reach ``variance_threshold`` in each case.
    common_matrix_ : np.ndarray of shape (n_input_channels, n_input_channels)
        Sum of the per-case leading-component projection matrices.

    See Also
    --------
    CLeVerCluster
        Select the channel nearest each DCPC-loading cluster centroid.
    CLeVerHybrid
        Select the highest-scoring channel within each DCPC-loading cluster.

    Notes
    -----
    CLeVer-Rank is unsupervised; any ``y`` passed to ``fit`` is ignored. Correlations
    are computed after centering each channel within each case. A channel that is
    constant to numerical precision contributes a zero row, column, and diagonal to
    that case's correlation matrix. If every channel in a case is constant, fitting
    raises ``ValueError``.

    References
    ----------
    .. [1] H. Yoon, K. Yang, and C. Shahabi, "Feature Subset Selection and
       Feature Ranking for Multivariate Time Series," IEEE Transactions on
       Knowledge and Data Engineering, 17(9), 1186--1198, 2005.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.channel_selection import CLeVerRank
    >>> X = np.random.default_rng(0).normal(size=(10, 6, 20))
    >>> Xt = CLeVerRank(n_channels=3).fit_transform(X)
    >>> Xt.shape
    (10, 3, 20)
    """

    def __init__(self, n_channels=5, variance_threshold=0.8):
        super().__init__(n_channels=n_channels, variance_threshold=variance_threshold)

    def _select_channels(self):
        self.channel_scores_ = np.linalg.norm(self.dcpc_, axis=0)
        self.channel_ranking_ = _stable_descending_order(self.channel_scores_)
        self.channels_selected_ = sorted(
            self.channel_ranking_[: self.n_channels].tolist()
        )


class CLeVerCluster(_BaseCLeVerClustering):
    """Select one channel nearest each cluster of CLeVer DCPC loadings.

    CLeVer-Cluster computes the shared CLeVer DCPC representation, treats the loading
    vector of each original channel as an observation for k-means, and selects the
    original channel nearest each fitted cluster centroid.

    Parameters
    ----------
    n_channels : int, default=5
        Exact number of original channels, and k-means clusters, to retain.
    variance_threshold : float, default=0.8
        Minimum cumulative explained-variance ratio used to determine the number of
        leading principal components for each case. Must be in ``(0, 1]``.
    n_init : int, default=20
        Number of k-means initializations. The result with lowest inertia is retained.
    max_iter : int, default=300
        Maximum number of k-means iterations for each initialization.
    tol : float, default=1e-4
        Nonnegative convergence tolerance used by k-means.
    random_state : int, RandomState instance or None, default=None
        Random state passed to k-means.

    Attributes
    ----------
    channels_selected_ : list[int]
        Selected channel indices in their original input order.
    dcpc_ : np.ndarray of shape (n_components_, n_input_channels)
        Descriptive common principal components stored as rows.
    n_components_ : int
        Maximum number of components required by any individual case.
    case_n_components_ : np.ndarray of shape (n_cases,)
        Number of components required to reach ``variance_threshold`` in each case.
    common_matrix_ : np.ndarray of shape (n_input_channels, n_input_channels)
        Sum of the per-case leading-component projection matrices.
    channel_loadings_ : np.ndarray of shape (n_input_channels, n_components_)
        DCPC loading vectors clustered by k-means.
    cluster_labels_ : np.ndarray of shape (n_input_channels,)
        Fitted cluster label for every original channel.
    cluster_centers_ : np.ndarray of shape (n_channels, n_components_)
        Fitted k-means cluster centers.
    inertia_ : float
        Sum of squared Euclidean distances to the closest fitted cluster center.

    See Also
    --------
    CLeVerRank
        Rank channels globally by their DCPC loading norms.
    CLeVerHybrid
        Select the highest-scoring channel within each loading cluster.

    Notes
    -----
    CLeVer-Cluster is unsupervised; any ``y`` passed to ``fit`` is ignored. It uses
    standard scikit-learn k-means inertia (squared Euclidean distances), whereas the
    paper describes a sum of unsquared Euclidean distances. Exact representative ties
    are resolved by lower original channel index.

    Correlations are computed after centering each channel within each case. A channel
    constant to numerical precision contributes a zero row, column, and diagonal in
    that case. Fitting raises ``ValueError`` if every channel in a case is constant.

    References
    ----------
    .. [1] H. Yoon, K. Yang, and C. Shahabi, "Feature Subset Selection and
       Feature Ranking for Multivariate Time Series," IEEE Transactions on
       Knowledge and Data Engineering, 17(9), 1186--1198, 2005.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.channel_selection import CLeVerCluster
    >>> X = np.random.default_rng(0).normal(size=(10, 6, 20))
    >>> Xt = CLeVerCluster(n_channels=3, random_state=0).fit_transform(X)
    >>> Xt.shape
    (10, 3, 20)
    """

    def __init__(
        self,
        n_channels=5,
        variance_threshold=0.8,
        n_init=20,
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        super().__init__(
            n_channels=n_channels,
            variance_threshold=variance_threshold,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def _select_channels(self):
        self._cluster_channels()
        selected = []
        for cluster_index in range(self.n_channels):
            members = np.flatnonzero(self.cluster_labels_ == cluster_index)
            distances = np.linalg.norm(
                self.channel_loadings_[members] - self.cluster_centers_[cluster_index],
                axis=1,
            )
            representative = members[np.lexsort((members, distances))[0]]
            selected.append(int(representative))
        self.channels_selected_ = sorted(selected)


class CLeVerHybrid(_BaseCLeVerClustering):
    """Select the highest-ranked channel within each CLeVer loading cluster.

    CLeVer-Hybrid shares the DCPC and k-means stages of CLeVer-Cluster, but selects
    from each cluster the channel with the largest CLeVer-Rank loading norm instead
    of the channel nearest the cluster centroid.

    Parameters
    ----------
    n_channels : int, default=5
        Exact number of original channels, and k-means clusters, to retain.
    variance_threshold : float, default=0.8
        Minimum cumulative explained-variance ratio used to determine the number of
        leading principal components for each case. Must be in ``(0, 1]``.
    n_init : int, default=20
        Number of k-means initializations. The result with lowest inertia is retained.
    max_iter : int, default=300
        Maximum number of k-means iterations for each initialization.
    tol : float, default=1e-4
        Nonnegative convergence tolerance used by k-means.
    random_state : int, RandomState instance or None, default=None
        Random state passed to k-means.

    Attributes
    ----------
    channels_selected_ : list[int]
        Selected channel indices in their original input order.
    channel_scores_ : np.ndarray of shape (n_input_channels,)
        Euclidean norm of each channel's DCPC loading vector.
    dcpc_ : np.ndarray of shape (n_components_, n_input_channels)
        Descriptive common principal components stored as rows.
    n_components_ : int
        Maximum number of components required by any individual case.
    case_n_components_ : np.ndarray of shape (n_cases,)
        Number of components required to reach ``variance_threshold`` in each case.
    common_matrix_ : np.ndarray of shape (n_input_channels, n_input_channels)
        Sum of the per-case leading-component projection matrices.
    channel_loadings_ : np.ndarray of shape (n_input_channels, n_components_)
        DCPC loading vectors clustered by k-means.
    cluster_labels_ : np.ndarray of shape (n_input_channels,)
        Fitted cluster label for every original channel.
    cluster_centers_ : np.ndarray of shape (n_channels, n_components_)
        Fitted k-means cluster centers.
    inertia_ : float
        Sum of squared Euclidean distances to the closest fitted cluster center.

    See Also
    --------
    CLeVerRank
        Rank channels globally by their DCPC loading norms.
    CLeVerCluster
        Select the channel nearest each loading-cluster centroid.

    Notes
    -----
    CLeVer-Hybrid is unsupervised; any ``y`` passed to ``fit`` is ignored. It uses
    standard scikit-learn k-means inertia (squared Euclidean distances), whereas the
    paper describes a sum of unsquared Euclidean distances. Exact score ties within a
    cluster are resolved by lower original channel index.

    Correlations are computed after centering each channel within each case. A channel
    constant to numerical precision contributes a zero row, column, and diagonal in
    that case. Fitting raises ``ValueError`` if every channel in a case is constant.

    References
    ----------
    .. [1] H. Yoon, K. Yang, and C. Shahabi, "Feature Subset Selection and
       Feature Ranking for Multivariate Time Series," IEEE Transactions on
       Knowledge and Data Engineering, 17(9), 1186--1198, 2005.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.channel_selection import CLeVerHybrid
    >>> X = np.random.default_rng(0).normal(size=(10, 6, 20))
    >>> Xt = CLeVerHybrid(n_channels=3, random_state=0).fit_transform(X)
    >>> Xt.shape
    (10, 3, 20)
    """

    def __init__(
        self,
        n_channels=5,
        variance_threshold=0.8,
        n_init=20,
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        super().__init__(
            n_channels=n_channels,
            variance_threshold=variance_threshold,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def _select_channels(self):
        self.channel_scores_ = np.linalg.norm(self.dcpc_, axis=0)
        self._cluster_channels()
        selected = []
        for cluster_index in range(self.n_channels):
            members = np.flatnonzero(self.cluster_labels_ == cluster_index)
            representative = members[
                np.lexsort((members, -self.channel_scores_[members]))[0]
            ]
            selected.append(int(representative))
        self.channels_selected_ = sorted(selected)
