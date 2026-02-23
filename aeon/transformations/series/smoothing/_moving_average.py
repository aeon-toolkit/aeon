"""Moving average transformation."""

__maintainer__ = ["Datadote"]
__all__ = ["MovingAverage"]

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class MovingAverage(BaseSeriesTransformer):
    """Calculate the moving average for a time series.

    Slides a window across the input array, and returns the averages for each window.

    Parameters
    ----------
    window_size: int, default=5
        Number of values to average for each window.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series.smoothing import MovingAverage
    >>> X = np.array([-3, -2, -1,  0,  1,  2,  3])
    >>> transformer = MovingAverage(2)
    >>> transformer.fit_transform(X)
    array([[-2.5, -1.5, -0.5,  0.5,  1.5,  2.5]])
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size

        super().__init__(axis=0)

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
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        csum = np.cumsum(X, axis=0)
        csum[self.window_size :, :] = (
            csum[self.window_size :, :] - csum[: -self.window_size, :]
        )
        Xt = csum[self.window_size - 1 :, :] / self.window_size
        return Xt
