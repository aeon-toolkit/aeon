"""Difference Transformer."""

import numpy as np

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
