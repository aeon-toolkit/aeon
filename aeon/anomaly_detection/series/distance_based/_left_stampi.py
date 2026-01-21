"""LeftSTAMPi anomaly detector."""

__maintainer__ = ["ferewi"]
__all__ = ["LeftSTAMPi"]


import numpy as np

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.utils.windowing import reverse_windowing


class LeftSTAMPi(BaseSeriesAnomalyDetector):
    """LeftSTAMPi anomaly detector.

    LeftSTAMPi [1]_ calculates the left matrix profile of a time series,
    which is the distance to the nearest neighbor of all already observed
    subsequences (i.e. all preceding subsequences) in the time series,
    in an incremental manner. The matrix profile is then used to calculate
    the anomaly score for each time point. The larger the distance to the
    nearest neighbor, the more anomalous the time point is.

    LeftSTAMPi supports univariate time series only.

    .. versionchanged:: 1.4.0
        **Breaking Change**: The ``predict(X)`` method now returns anomaly scores
        of length ``len(X)`` (matching the input length), rather than scores for
        the entire accumulated time series. This fixes issue #2819 where the output
        shape was inconsistent with the input.

        **Impact on existing code**: If your code expects ``predict(X)`` to return
        scores for all points seen since ``fit()``, you will need to accumulate
        results yourself or use ``fit_predict()`` on the complete series.

        **Migration example**:

        .. code-block:: python

            # Old behavior (before v1.4.0):
            detector.fit(X_train)
            all_scores = detector.predict(X_test)  # Returns len(X_train) + len(X_test)

            # New behavior (v1.4.0+):
            detector.fit(X_train)
            test_scores = detector.predict(X_test)  # Returns len(X_test)
            # To get all scores:
            all_scores = detector.fit_predict(np.concatenate([X_train, X_test]))

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

    Examples
    --------
    Calculate the anomaly score for the complete time series at once.
    Internally,this is applying the incremental approach outlined below.

    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.distance_based import LeftSTAMPi
    >>> X = np.random.default_rng(42).random((10))  # doctest: +SKIP
    >>> detector = LeftSTAMPi(window_size=3, n_init_train=3)  # doctest: +SKIP
    >>> detector.fit_predict(X)  # doctest: +SKIP
    array([0.        , 0.        , 0.        , 0.07042306, 0.15989868,
           0.68912499, 0.75398303, 0.89696118, 0.5516023 , 0.69736132])

    Incremental prediction example (showing v1.4.0+ behavior):

    >>> X_train = np.random.default_rng(42).random((10))  # doctest: +SKIP
    >>> X_test = np.random.default_rng(43).random((5))  # doctest: +SKIP
    >>> detector = LeftSTAMPi(window_size=3, n_init_train=3)  # doctest: +SKIP
    >>> detector.fit(X_train)  # doctest: +SKIP
    >>> test_scores = detector.predict(X_test)  # doctest: +SKIP
    >>> len(test_scores) == len(X_test)  # True in v1.4.0+  # doctest: +SKIP
    True

    References
    ----------
    .. [1] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum,
           Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen,
           and Eamonn Keogh: "Matrix Profile I: All Pairs Similarity Joins
           for Time Series: A Unifying View That Includes Motifs, Discords
           and Shapelets.", In Proceedings of the International Conference
           on Data Mining (ICDM), 1317–1322. doi: 10.1109/ICDM.2016.0179

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

    def _check_params_fit(self, X):
        """Validate parameters for initial fit operation."""
        if self.window_size < 3 or self.window_size > len(X):
            raise ValueError(
                "The window size must be at least 3 and "
                "at most the length of the time series."
            )

        if len(X) < self.window_size:
            raise ValueError(
                f"The time series length ({len(X)}) must be at least "
                f"window_size ({self.window_size}) for initial fitting."
            )

        if self.window_size > self.n_init_train:
            raise ValueError(
                f"The window size ({self.window_size}) must be less than or equal to "
                f"n_init_train ({self.n_init_train})"
            )

        if self.k < 1 or self.k > len(X) - self.window_size + 1:
            raise ValueError(
                "The top `k` distances must be at least 1 and at most the length of "
                "the time series minus the window size."
            )

    def _check_params_predict(self, X):
        """Validate parameters for incremental predict operation.

        For predict(), X can be any length >= 1 since it's being added to
        an existing matrix profile initialized during fit().
        """
        # X should already be ensured to be an array in _predict
        # No strict validation needed for incremental updates
        pass

    def _fit(self, X: np.ndarray, y=None) -> "LeftSTAMPi":
        if X.ndim > 1:
            X = X.squeeze()
        self._check_params_fit(X)

        self._call_stumpi(X)

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 1:
            X = X.squeeze()

        # Ensure X is always 1D array (squeeze() can turn single element into scalar)
        X = np.atleast_1d(X)

        self._check_params_predict(X)

        n_new_points = len(X)
        if n_new_points == 0:
            return np.array([])

        # Store initial matrix profile length before updating
        initial_mp_len = len(self.mp_._left_P)

        # Update matrix profile incrementally with each new point
        for x in X:
            self.mp_.update(x)

        # Extract only the newly added portion of the matrix profile
        # Each new point adds one new window to the matrix profile
        new_mp_windows = self.mp_._left_P[initial_mp_len:]

        # reverse_windowing converts window scores back to point scores
        # Formula: n_timepoints = (n_windows - 1) * stride + window_size
        # For n_new_points, we add n_new_points windows
        point_anomaly_scores = reverse_windowing(new_mp_windows, self.window_size)

        # reverse_windowing may produce window_size - 1 extra points at boundaries
        # Trim to exactly match the number of new input points
        expected_length = n_new_points
        actual_length = len(point_anomaly_scores)

        if actual_length > expected_length:
            # Trim excess points from the end (boundary effect from reverse_windowing)
            point_anomaly_scores = point_anomaly_scores[:expected_length]
        elif actual_length < expected_length:
            # This should not happen with correct stumpy behavior
            # Pad with zeros if needed for robustness
            point_anomaly_scores = np.pad(
                point_anomaly_scores,
                (0, expected_length - actual_length),
                mode="constant",
                constant_values=0,
            )

        return point_anomaly_scores

    def _fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        if X.ndim > 1:
            X = X.squeeze()

        n_total_points = len(X)
        if n_total_points < self.n_init_train:
            raise ValueError(
                f"Input length ({n_total_points}) must be at least "
                f"n_init_train ({self.n_init_train})"
            )

        # Initialize with first n_init_train points
        self.fit(X[: self.n_init_train])

        # Predict on remaining points
        if n_total_points > self.n_init_train:
            test_scores = self.predict(X[self.n_init_train :])
        else:
            test_scores = np.array([])

        # For fit_predict, return scores for the entire series
        # First n_init_train points get zero scores (no left neighbors)
        full_scores = np.zeros(n_total_points, dtype=np.float64)

        # Assign test scores to the appropriate positions
        n_test_points = n_total_points - self.n_init_train
        if n_test_points > 0:
            # Ensure test_scores has correct length before assignment
            if len(test_scores) != n_test_points:
                raise ValueError(
                    f"predict() returned {len(test_scores)} scores, "
                    f"expected {n_test_points}"
                )
            full_scores[self.n_init_train :] = test_scores

        return full_scores

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
