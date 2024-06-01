"""Implements an adapter for PyOD models to be used in the Aeon framework."""

from __future__ import annotations

__maintainer__ = ["CodeLionX"]
__all__ = ["PyODAdapter"]

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from aeon.anomaly_detection.base import BaseAnomalyDetector

if TYPE_CHECKING:
    from pyod.models.base import BaseDetector


class PyODAdapter(BaseAnomalyDetector):
    """Adapter for PyOD anomaly detection models to be used in the Aeon framework.

    This adapter allows the use of PyOD models in the Aeon framework. The adapter
    takes a PyOD model and applies it to a sliding window of the input data. The
    anomaly score of each window is then averaged to obtain the final anomaly score
    for each data instance. If the window size is set to 1, the adapter applies the
    PyOD model to each data instance individually resembling the original behavior of
    the PyOD model. If the striding size is set to the window size, the adapter
    creates tumbling windows (non-overlapping) instead of sliding windows. The anomaly
    score for each data point is, then, computed based on the score of the single
    tumbling window containing the data point.

    Both univariate and multivariate time series are supported. For multivariate time
    series the adapter concatenates the data points of each channel in the window to
    a single univariate feature vector per window as input to the PyOD model.

    Parameters
    ----------
    pyod_model : BaseDetector
        Instance of a PyOD anomaly detection model used for the detection.
    window_size : int, default=10
        Size of the sliding window.
    stride : int, default=1
        Stride of the sliding window.

    Examples
    --------
    >>> import numpy as np
    >>> from pyod.models.lof import LOF
    >>> from aeon.anomaly_detection import PyODAdapter
    >>> X = np.random.default_rng(42).random((10, 2), dtype=np.float_)
    >>> detector = PyODAdapter(LOF(), window_size=2)
    >>> detector.fit_predict(X, axis=0)
    array([1.02352234 1.00193038 0.98584441 0.99630753 1.00656619 1.00682081 1.00781515
           0.99709741 0.98878895 0.99723947])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": True,
        "capability:missing_values": False,
        # Omit the version specification until PyOD has __version__
        # (https://github.com/yzhao062/pyod/pull/584)
        # "python_dependencies": ["pyod>=1.1.3"]
        "python_dependencies": ["pyod"],
    }

    def __init__(
        self, pyod_model: BaseDetector, window_size: int = 10, stride: int = 1
    ):
        self.pyod_model = pyod_model
        self.window_size = window_size
        self.stride = stride

        self._padding_length = 0
        super().__init__(axis=0)

    @staticmethod
    def _is_pyod_model(model: Any) -> bool:
        """Check if the provided model is a PyOD model."""
        from pyod.models.base import BaseDetector

        # allow duck-typing for tests?
        def looks_like_pyod_model(model: Any) -> bool:
            return hasattr(model, "fit") and hasattr(model, "decision_scores_")

        return isinstance(model, BaseDetector) or looks_like_pyod_model(model)

    def _predict(self, X) -> np.ndarray:
        if not self._is_pyod_model(self.pyod_model):
            raise ValueError("The provided model is not a compatible PyOD model.")

        if self.window_size < 1 or self.window_size > X.shape[0]:
            raise ValueError(
                "The window size must be at least 1 and at most the length of the "
                "time series."
            )

        if self.stride < 1 or self.stride > self.window_size:
            raise ValueError(
                "The stride must be at least 1 and at most the window size."
            )

        _X = self._preprocess_data(X)
        self.pyod_model.fit(_X)
        scores = self.pyod_model.decision_scores_
        scores = self._postprocess_scores(scores)
        return scores

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        flat_shape = (
            X.shape[0] - (self.window_size - 1),
            -1,
        )  # in case we have a multivariate TS
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(
            flat_shape
        )[:: self.stride, :]
        self._padding_length = X.shape[0] - (
            slides.shape[0] * self.stride + self.window_size - self.stride
        )
        return slides

    def _postprocess_scores(self, scores: np.ndarray) -> np.ndarray:
        # compute begin and end indices of windows
        begins = np.array([i * self.stride for i in range(scores.shape[0])])
        ends = begins + self.window_size

        # prepare target array
        unwindowed_length = (
            self.stride * (scores.shape[0] - 1)
            + self.window_size
            + self._padding_length
        )
        mapped = np.full(unwindowed_length, fill_value=np.nan)

        # only iterate over window intersections
        indices = np.unique(np.r_[begins, ends])
        for i, j in zip(indices[:-1], indices[1:]):
            window_indices = np.flatnonzero((begins <= i) & (j - 1 < ends))
            mapped[i:j] = np.nanmean(scores[window_indices])

        # replace untouched indices with 0 (especially for the padding at the end)
        np.nan_to_num(mapped, copy=False)
        return mapped

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {"pyod_model": _PyODMock(), "window_size": 5, "stride": 1}


class _PyODMock:
    def __init__(self, anomaly_position: int | None = None):
        self._anomaly_position = anomaly_position
        self.decision_scores_ = np.empty(0, dtype=np.float_)

    def fit(self, X, y=None):
        self._X = X
        self._y = y
        self.decision_scores_ = self._decision_scores_(X.shape[0])

    def _decision_scores_(self, n):
        scores = np.zeros(n, dtype=np.float_)
        if self._anomaly_position is not None:
            scores[self._anomaly_position] = 1.0
        return scores
