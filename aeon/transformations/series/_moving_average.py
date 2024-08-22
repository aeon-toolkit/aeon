"""Moving average transformation."""

__maintainer__ = ["Datadote"]
__all__ = "MovingAverageTransformer"

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class MovingAverageTransformer(BaseSeriesTransformer):
    """Filter a time series using a simple moving average.

    Parameters
    ----------
    window_size: int, default=5
        Number of values to average for each window

    References
    ----------
    Large, J., Southam, P., Bagnall, A. (2019).
        Can Automated Smoothing Significantly Improve Benchmark Time Series
        Classification Algorithms?. In: Pérez García, H., Sánchez González,
        L., CastejónLimas, M., Quintián Pardo, H., Corchado Rodríguez, E. (eds) Hybrid
        Artificial Intelligent Systems. HAIS 2019. Lecture Notes in Computer Science(),
        vol 11734. Springer, Cham. https://doi.org/10.1007/978-3-030-29859-3_5
        https://link.springer.com/chapter/10.1007/978-3-030-29859-3_5
        https://arxiv.org/abs/1811.00894

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._moving_average import MovingAverageTransformer
    >>> X = np.array([-3, -2, -1,  0,  1,  2,  3])
    >>> transformer = MovingAverageTransformer(2)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (1, 6)
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, window_size: int = 5) -> None:
        super().__init__(axis=0)
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self.window_size = window_size

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
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        csum = np.cumsum(X, axis=0)
        csum[self.window_size :, :] = (
            csum[self.window_size :, :] - csum[: -self.window_size, :]
        )
        Xt = csum[self.window_size - 1 :, :] / self.window_size
        return Xt
