"""Exponential smoothing transformation."""

__maintainer__ = ["Datadote"]
__all__ = ["ExponentialSmoothing"]


import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class ExponentialSmoothing(BaseSeriesTransformer):
    """Filter a time series using exponential smoothing.

    - Exponential smoothing (EXP) is a generalisaton of moving average smoothing that
    assigns a decaying weight to each element rather than averaging over a window.
    - Assume time series T = [t_0, ..., t_j], and smoothed values S = [s_0, ..., s_j]
    - Then, s_0 = t_0 and s_j = alpha * t_j + (1 - alpha) * s_j-1
    where 0 ≤ alpha ≤ 1. If window_size is given, alpha is overwritten, and set as
    alpha = 2. / (window_size + 1)

    Parameters
    ----------
    alpha: float, default=0.2
        decaying weight. Range [0, 1]. Overwritten by window_size if window_size exists
    window_size: int or float or None, default=None
        If window_size is specified, alpha is set to 2. / (window_size + 1)

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series.smoothing import ExponentialSmoothing
    >>> X = np.array([-2, -1,  0,  1,  2])
    >>> transformer = ExponentialSmoothing(0.5)
    >>> transformer.fit_transform(X)
    array([[-2.    , -1.5   , -0.75  ,  0.125 ,  1.0625]])
    >>> X = np.array([[1, 2, 3, 4], [10, 9, 8, 7]])
    >>> transformer.fit_transform(X)
    array([[ 1.   ,  1.5  ,  2.25 ,  3.125],
           [10.   ,  9.5  ,  8.75 ,  7.875]])
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(
        self, alpha: float = 0.2, window_size: int | float | None = None
    ) -> None:
        self.alpha = alpha if window_size is None else 2.0 / (window_size + 1)
        self.window_size = window_size

        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt: 2D np.ndarray
            transformed version of X
        """
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in range [0, 1], got {self.alpha}")
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        Xt = np.zeros_like(X, dtype="float")
        Xt[:, 0] = X[:, 0]
        for i in range(1, Xt.shape[1]):
            Xt[:, i] = self.alpha * X[:, i] + (1 - self.alpha) * Xt[:, i - 1]
        return Xt
