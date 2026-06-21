"""Nearest centroid classifier for time series."""

__maintainer__ = []
__all__ = ["NearestCentroidClassifier"]

import numpy as np

from aeon.classification.base import BaseClassifier
from aeon.clustering.averaging import elastic_barycenter_average, mean_average
from aeon.clustering.averaging._ba_utils import VALID_SOFT_BA_METHODS
from aeon.distances import pairwise_distance
from aeon.utils.validation import check_n_jobs


class NearestCentroidClassifier(BaseClassifier):
    """Nearest centroid (Rocchio) classifier for time series.

    Computes a single centroid per class by averaging that class's training
    series, then classifies a new series by assigning it to the nearest class
    centroid under a chosen (elastic) distance.

    The centroid is the elastic barycentre of the class series. For an ordinary
    distance this is a discrete barycentre average (DTW Barycentre Averaging,
    DBA, when ``distance="dtw"``); for a soft distance (``"soft_dtw"`` /
    ``"soft_msm"``) it is the gradient-based soft barycentre. ``"mean"`` uses a
    plain arithmetic mean.

    Parameters
    ----------
    distance : str, default="dtw"
        Distance used to assign test series to the nearest centroid and, for
        barycentre averaging, to compute the centroids. See
        :func:`aeon.distances.pairwise_distance` for valid options.
    average_method : str or None, default=None
        How class centroids are computed. One of ``"mean"``, ``"petitjean"``,
        ``"subgradient"``, ``"kasba"`` or ``"soft"``. If ``None`` (default), it
        resolves to ``"soft"`` for a soft ``distance`` and ``"petitjean"`` (DBA)
        otherwise. A soft ``distance`` requires ``average_method="soft"`` (the
        default resolves to it automatically); pairing a soft distance with a
        non-soft averaging method, or vice versa, raises ``ValueError``.
    distance_params : dict or None, default=None
        Keyword arguments for the distance (e.g. ``{"window": 0.2}`` for DTW or
        ``{"gamma": 0.1}`` for soft distances), used for both averaging and
        nearest-centroid assignment.
    average_params : dict or None, default=None
        Keyword arguments forwarded to the barycentre averaging (e.g.
        ``{"max_iters": 50}``). Ignored when ``average_method="mean"``.
    n_jobs : int, default=1
        The number of jobs to run in parallel. ``-1`` means using all
        processors.

    Attributes
    ----------
    classes_ : np.ndarray
        The class labels, ordered as the centroids.
    centroids_ : np.ndarray of shape (n_classes, n_channels, n_timepoints)
        The per-class centroids.

    See Also
    --------
    KNeighborsTimeSeriesClassifier : Distance-based nearest neighbours classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.classification.distance_based import NearestCentroidClassifier
    >>> X = np.array([[[1.0, 2, 3, 4, 5]], [[1.0, 2, 3, 4, 6]],
    ...               [[8.0, 7, 6, 5, 4]], [[8.0, 7, 6, 5, 3]]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = NearestCentroidClassifier(
    ...     distance="euclidean", average_method="mean"
    ... ).fit(X, y)
    >>> clf.predict(np.array([[[1.0, 2, 3, 4, 5]]]))
    array([0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "distance",
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        distance: str = "dtw",
        average_method: str | None = None,
        distance_params: dict | None = None,
        average_params: dict | None = None,
        n_jobs: int = 1,
    ):
        self.distance = distance
        self.average_method = average_method
        self.distance_params = distance_params
        self.average_params = average_params
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y):
        self._check_params()
        self.classes_ = np.unique(y)
        self.centroids_ = np.zeros((len(self.classes_), X.shape[1], X.shape[2]))

        for i, label in enumerate(self.classes_):
            class_X = X[y == label]
            if self._average_method == "mean":
                self.centroids_[i] = mean_average(class_X)
            else:
                self.centroids_[i] = elastic_barycenter_average(
                    class_X,
                    distance=self.distance,
                    method=self._average_method,
                    n_jobs=self._n_jobs,
                    **self._average_params,
                    **self._distance_params,
                )
        return self

    def _predict(self, X) -> np.ndarray:
        pairwise_matrix = pairwise_distance(
            X,
            self.centroids_,
            method=self.distance,
            n_jobs=self._n_jobs,
            **self._distance_params,
        )
        return self.classes_[pairwise_matrix.argmin(axis=1)]

    def _check_params(self):
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._distance_params = self.distance_params or {}
        self._average_params = self.average_params or {}

        is_soft = self.distance in VALID_SOFT_BA_METHODS
        if self.average_method is None:
            self._average_method = "soft" if is_soft else "petitjean"
        else:
            self._average_method = self.average_method
            if is_soft and self._average_method != "soft":
                raise ValueError(
                    f"distance={self.distance!r} is a soft distance and can only "
                    "be averaged with average_method='soft', got "
                    f"average_method={self.average_method!r}."
                )
            if not is_soft and self._average_method == "soft":
                raise ValueError(
                    "average_method='soft' requires a soft distance, one of "
                    f"{VALID_SOFT_BA_METHODS}, got distance={self.distance!r}."
                )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return testing parameter settings for the estimator."""
        return {"distance": "euclidean", "average_method": "mean"}
