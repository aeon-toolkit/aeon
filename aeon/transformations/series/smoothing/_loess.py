"""LOESS Smoothing (Locally Estimated Scatterplot Smoothing)."""

__maintainer__ = []
__all__ = ["LoessSmoother"]

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class LoessSmoother(BaseSeriesTransformer):
    """Locally Estimated Scatterplot Smoothing (LOESS).

    A non-parametric regression technique that fits a smooth curve through a
    time series. For each point in the series, a polynomial is fitted to a
    subset of the data (local window) using weighted least squares.

    Parameters
    ----------
    span : float, default=0.5
        The fraction of the data to use for the local window.
        Must be between 0 and 1. For example, 0.5 means 50% of the data
        is used for each local fit.
    degree : int, default=1
        The degree of the local polynomial to fit.
        1 = Linear (standard LOESS), 2 = Quadratic.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series.smoothing import LoessSmoother
    >>> X = np.array([-3, -2, -1,  0,  1,  2,  3])
    >>> transformer = LoessSmoother(span=0.5, degree=1)
    >>> transformer.fit_transform(X)

    References
    ----------
    [1] Cleveland, W. S. (1979). Robust locally weighted regression and
       smoothing scatterplots. Journal of the American statistical association,
       74(368), 829-836.
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, span: float = 0.5, degree: int = 1) -> None:
        self.span = span
        self.degree = degree
        super().__init__(axis=1)

        if not (0 < self.span <= 1):
            raise ValueError(f"span must be in (0, 1], but got {self.span}")

        if self.degree not in [1, 2]:
            raise ValueError(f"degree must be 1 or 2, but got {self.degree}")

    def _transform(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_channels, n_timepoints = X.shape
        Xt = np.zeros_like(X, dtype=float)

        if n_timepoints <= self.degree:
            return X

        k = int(np.ceil(self.span * n_timepoints))
        k = max(2, min(k, n_timepoints))

        for i in range(n_timepoints):
            distances = np.abs(np.arange(n_timepoints) - i)
            sorted_indices = np.argpartition(distances, k - 1)[:k]

            local_dists = distances[sorted_indices]

            d_max = local_dists.max()
            if d_max == 0:
                d_max = 1.0

            weights = (1 - (local_dists / d_max) ** 3) ** 3
            weights[weights < 0] = 0

            local_t = sorted_indices - i

            H = np.vander(local_t, N=self.degree + 1, increasing=True)
            W = np.diag(weights)
            A = H.T @ W @ H

            for c in range(n_channels):
                local_y = X[c, sorted_indices]
                b = H.T @ W @ local_y

                beta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                Xt[c, i] = beta[0]

        return Xt
