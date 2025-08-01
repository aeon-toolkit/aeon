"""Difference Transformer."""

import numpy as np
from numba import njit

from aeon.transformations.series.base import BaseSeriesTransformer

__maintainer__ = ["TinaJin0228", "alexbanwell1"]
__all__ = ["DifferenceTransformer"]


class DifferenceTransformer(BaseSeriesTransformer):
    """
    Calculates the n-th order difference of a time series.

    Transforms a time series X into a series Y representing the difference
    calculated `order` times.

    The time series are supposed to be all in rows,
    with shape (n_channels, n_timepoints)

    - Order 1: Y[t] = X[t] - X[t-1]
    - Order 2: Y[t] = (X[t] - X[t-1]) - (X[t-1] - X[t-2]) = X[t] - 2*X[t-1] + X[t-2]
    - ... and so on.

    The transformed series will be shorter than the input series by `order`
    elements along the time axis.

    Parameters
    ----------
    order : int, default=1
        The order of differencing. Must be a positive integer.

    Notes
    -----
    This transformer assumes the input series does not contain NaN values where
    the difference needs to be computed.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._diff import DifferenceTransformer
    >>> X1 = np.array([[1, 3, 2, 5, 4, 7, 6, 9, 8, 10]]) # Shape (1, 10)
    >>> dt = DifferenceTransformer()
    >>> Xt1 = dt.fit_transform(X1)
    >>> print(Xt1) # Shape (1, 9)
    [[ 2 -1  3 -1  3 -1  3 -1  2]]

    >>> X2 = np.array([[1, 3, 2, 5, 4, 7, 6, 9, 8, 10]]) # Shape (1, 10)
    >>> dt2 = DifferenceTransformer(order=2)
    >>> Xt2 = dt2.fit_transform(X2)
    >>> print(Xt2) # Shape (1, 8)
    [[-3  4 -4  4 -4  4 -4  3]]

    >>> X3 = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]) # Shape (2, 5)
    >>> dt = DifferenceTransformer()
    >>> Xt3 = dt.fit_transform(X3)
    >>> print(Xt3) # Shape (2, 4)
    [[ 1  1  1  1]
     [-1 -1 -1 -1]]
    """

    _tags = {
        "capability:multivariate": True,
        "capability:inverse_transform": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, order=1):
        self.order = order
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """
        Perform the n-th order differencing transformation.

        Parameters
        ----------
        X : Time series to transform. With shape (n_channels, n_timepoints).
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : np.ndarray
        """
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError(
                f"`order` must be a positive integer, but got {self.order}"
            )

        diff_X = np.diff(X, n=self.order, axis=1)

        Xt = diff_X

        return Xt

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform to reconstruct the original time series.

        Parameters
        ----------
        X : Time series to inverse transform. With shape (n_channels, n_timepoints).
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : np.ndarray
            Reconstructed original time series.
        """
        if y is None or y.shape[1] < self.order:
            raise ValueError(
                f"Inverse transformm requires first {self.order} original \
                  data values supplied as y, but inverse_transform called with y=None"
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have the same number of channels as X. "
                f"Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        initial_values = y[:, : self.order]

        return np.array(
            [
                _undifference(diff_X, initial_value)
                for diff_X, initial_value in zip(X, initial_values)
            ]
        )


@njit(cache=True, fastmath=True)
def _comb(n, k):
    """
    Calculate the binomial coefficient C(n, k) = n! / (k! * (n - k)!).

    Parameters
    ----------
    n : int
        The total number of items.
    k : int
        The number of items to choose.

    Returns
    -------
    int
        The binomial coefficient C(n, k).
    """
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k  # Take advantage of symmetry
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


@njit(cache=True, fastmath=True)
def _undifference(diff, initial_values):
    """
    Reconstruct original time series from an n-th order differenced series.

    Parameters
    ----------
    diff : array-like
        n-th order differenced series of length N - n
    initial_values : array-like
        The first n values of the original series before differencing (length n)

    Returns
    -------
    original : np.ndarray
        Reconstructed original series of length N
    """
    n = len(initial_values)
    kernel = np.array(
        [(-1) ** (k + 1) * _comb(n, k) for k in range(1, n + 1)],
        dtype=initial_values.dtype,
    )
    original = np.empty((n + len(diff)), dtype=initial_values.dtype)
    original[:n] = initial_values

    for i, d in enumerate(diff):
        original[n + i] = np.dot(kernel, original[i : n + i][::-1]) + d

    return original
