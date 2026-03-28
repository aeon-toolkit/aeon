"""LeftSTAMPi anomaly detector."""

__maintainer__ = ["ferewi"]
__all__ = ["LeftSTAMPi"]


import numpy as np

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector


class LeftSTAMPi(BaseSeriesAnomalyDetector):
    """LeftSTAMPi anomaly detector.

    LeftSTAMPi [1]_ calculates the left matrix profile of a time series,
    which is the distance to the nearest neighbor of all already observed
    subsequences (i.e. all preceding subsequences) in the time series,
    in an incremental manner. The matrix profile is then used to calculate
    the anomaly score for each time point. The larger the distance to the
    nearest neighbor, the more anomalous the time point is.

    LeftSTAMPi supports univariate time series only.

    Parameters
    ----------
    window_size : int, default=3
        Size of the sliding window. Defaults to the minimal possible value of 3.
    n_init_train: int, default=3
        The number of points used to init the matrix profile.
        n_init_train must not be smaller than window_size.
        The discord will not be found in this segment.
    normalize : bool, default=True
        Whether to normalize the windows before computing the distance.
    p : float, default=2.0
        The p-norm to use for the distance calculation.
    k : int, default=1
        The number of top distances to return.

    Notes
    -----
    The first ``n_init_train`` points will always receive an anomaly score of 0,
    as there are no left neighbors available for comparison in that region.
    This applies to both ``fit_predict`` and ``predict`` calls.

    Examples
    --------
    Calculate the anomaly score for the complete time series at once.
    Internally, this is applying the incremental approach outlined below.

    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.distance_based import LeftSTAMPi
    >>> X = np.random.default_rng(42).random((10))  # doctest: +SKIP
    >>> detector = LeftSTAMPi(window_size=3, n_init_train=3)  # doctest: +SKIP
    >>> detector.fit_predict(X)  # doctest: +SKIP
    array([0.        , 0.        , 0.        , 0.07042306, 0.15989868,
           0.68912499, 0.75398303, 0.89696118, 0.5516023 , 0.69736132])

    References
    ----------
    .. [1] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum,
           Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen,
           and Eamonn Keogh: "Matrix Profile I: All Pairs Similarity Joins
           for Time Series: A Unifying View That Includes Motifs, Discords
           and Shapelets.", In Proceedings of the International Conference
           on Data Mining (ICDM), 1317-1322. doi: 10.1109/ICDM.2016.0179

    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "cant_pickle": True,
        "python_dependencies": ["stumpy"],
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
    }

    def __init__(
        self,
        window_size: int = 3,
        n_init_train: int = 3,
        normalize: bool = True,
        p: float = 2.0,
        k: int = 1,
    ):
        self.mp_: np.ndarray | None = None
        self.window_size = window_size
        self.n_init_train = n_init_train
        self.normalize = normalize
        self.p = p
        self.k = k

        super().__init__(axis=0)

    def _check_params(self, X):
        """Validate parameters against the fit data X."""
        if self.window_size < 3 or self.window_size > len(X):
            raise ValueError(
                "The window size must be at least 3 and at most the length of the "
                "time series."
            )

        if self.window_size > self.n_init_train:
            raise ValueError(
                f"The window size must be less than or equal to "
                f"n_init_train (is: {self.n_init_train})"
            )

        if self.k < 1 or self.k > len(X) - self.window_size + 1:
            raise ValueError(
                "The top `k` distances must be at least 1 and at most the length of "
                "the time series minus the window size."
            )

    def _fit(self, X: np.ndarray, y=None) -> "LeftSTAMPi":
        if X.ndim > 1:
            X = X.squeeze()

        self._check_params(X)

        # Initialise the matrix profile on only the first n_init_train points.
        # Store the full input length so _predict can detect the base-class
        # fit_predict path where _fit(X) and _predict(X) both receive full X.
        self._n_fit_points = len(X)
        self._call_stumpi(X[: self.n_init_train])

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 1:
            X = X.squeeze()

        n_input = len(X)

        # Base-class fit_predict path: _fit(full_X) then _predict(full_X).
        # Prepend zeros for the init region and score only the remainder.
        if hasattr(self, "_n_fit_points") and self._n_fit_points == n_input:
            init_scores = np.zeros(self.n_init_train, dtype=np.float64)
            n_test = n_input - self.n_init_train
            test_scores = self._score_new_points(X[self.n_init_train :], n_test)
            # Reset so subsequent standalone predict() calls work correctly
            self._n_fit_points = 0
            return np.concatenate([init_scores, test_scores])

        # Standalone predict() path: X contains only new (unseen) points
        return self._score_new_points(X, len(X))

    def _score_new_points(self, X: np.ndarray, n_out: int) -> np.ndarray:
        """Incrementally update the matrix profile and return ``n_out`` scores.

        Each call to ``mp_.update(point)`` appends one new window to the left
        matrix profile.  The score for each new point is assigned to the end
        of the window that completes at that point (i.e. window ``j`` maps to
        output index ``j + window_size - 1``).  The first ``window_size - 1``
        output positions, which have no completing window yet, receive a score
        of 0.

        Parameters
        ----------
        X : np.ndarray
            New data points to feed into the matrix profile incrementally.
        n_out : int
            Number of point-level scores to return (equals ``len(X)``).
        """
        if n_out == 0:
            return np.array([], dtype=np.float64)

        # Number of windows already in the profile before adding new points.
        # After fitting on n_init_train points: n_init_windows = n_init_train
        # - window_size + 1.
        n_init_windows = self.n_init_train - self.window_size + 1

        # Feed each new point into the existing stumpy STUMPI object.
        for point in X:
            self.mp_.update(point)

        # Extract only the newly added window scores from the left matrix
        # profile.  Each update call appends exactly one new entry.
        new_mp_windows = self.mp_._left_P[n_init_windows:]

        # Assign each window score to the last (rightmost) point it covers.
        # Window j of the new windows ends at output index j + window_size - 1.
        # Indices below window_size - 1 have no completing window and stay 0.
        n_total_out = len(new_mp_windows) + self.window_size - 1
        point_scores = np.zeros(n_total_out, dtype=np.float64)
        for j, score in enumerate(new_mp_windows):
            point_scores[j + self.window_size - 1] = score

        return point_scores[:n_out]

    def _call_stumpi(self, X: np.ndarray):
        import stumpy

        self.mp_ = stumpy.stumpi(
            X,
            m=self.window_size,
            egress=False,
            normalize=self.normalize,
            p=self.p,
            k=self.k,
        )