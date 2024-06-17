"""Implements an adapter for PyOD models to be used in the Aeon framework."""

from __future__ import annotations

__maintainer__ = ["CodeLionX"]
__all__ = ["PyODAdapter"]

from typing import TYPE_CHECKING, Any

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.windowing import reverse_windowing, sliding_windows

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

    .. list-table:: Capabilities
       :stub-columns: 1

       * - Input data format
         - univariate and multivariate
       * - Output data format
         - anomaly scores
       * - Learning Type
         - unsupervised


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
    >>> from pyod.models.lof import LOF  # doctest: +SKIP
    >>> from aeon.anomaly_detection import PyODAdapter  # doctest: +SKIP
    >>> X = np.random.default_rng(42).random((10, 2), dtype=np.float_)
    >>> detector = PyODAdapter(LOF(), window_size=2)  # doctest: +SKIP
    >>> detector.fit_predict(X, axis=0)  # doctest: +SKIP
    array([1.02352234 1.00193038 0.98584441 0.99630753 1.00656619 1.00682081 1.00781515
           0.99709741 0.98878895 0.99723947])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": True,
        "capability:missing_values": False,
        # Omit the version specification until PyOD has __version__
        # (https://github.com/yzhao062/pyod/pull/584 in dev but not released yet)
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

        return isinstance(model, BaseDetector)

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

        _X, self._padding_length = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        self.pyod_model.fit(_X)
        scores = self.pyod_model.decision_scores_
        scores = reverse_windowing(
            scores, self.window_size, np.nanmean, self.stride, self._padding_length
        )
        return scores

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
        _check_soft_dependencies("pyod")

        from pyod.models.lof import LOF

        return {"pyod_model": LOF(), "window_size": 5, "stride": 1}
