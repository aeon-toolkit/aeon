import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer

__maintainer__ = []
__all__ = ["DifferenceTransformer"]


class DifferenceTransformer(BaseSeriesTransformer):
    """
    Calculates the n-th order difference of a time series.

    Transforms a time series X into a series Y representing the difference
    calculated `order` times.
    - Order 1: Y[t] = X[t] - X[t-1]
    - Order 2: Y[t] = (X[t] - X[t-1]) - (X[t-1] - X[t-2]) = X[t] - 2*X[t-1] + X[t-2]
    - ... and so on.

    The first `order` element(s) of the transformed series along the time axis
    will be NaN, so that the output series will have the same shape as the input series.

    Parameters
    ----------
    order : int, default=1
        The order of differencing. Must be a positive integer.

    axis : int, default=1
        The axis along which the difference is computed. Assumed to be the
        time axis.
        If `axis == 0`, assumes shape `(n_timepoints, n_channels)`.
        If `axis == 1`, assumes shape `(n_channels, n_timepoints)`.

    Notes
    -----
    This transformer assumes the input series does not contain NaN values where
    the difference needs to be computed.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.series._diff import DifferenceTransformer
    >>> X1 = np.array([[1, 3, 2, 5, 4, 7, 6, 9, 8, 10]])
    >>> dt = DifferenceTransformer()
    >>> Xt1 = dt.fit_transform(X1)
    >>> print(Xt1)
    [[nan  2. -1.  3. -1.  3. -1.  3. -1.  2.]]

    >>> X2 = np.array([[1, 3, 2, 5, 4, 7, 6, 9, 8, 10]])
    >>> dt2 = DifferenceTransformer(order=2)
    >>> Xt2 = dt2.fit_transform(X2)
    >>> print(Xt2)
    [[nan nan -3.  4. -4.  4. -4.  4. -4.  3.]]

    >>> X3 = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    >>> dt = DifferenceTransformer()
    >>> Xt3 = dt.fit_transform(X3)
    >>> print(Xt3)
    [[nan  1.  1.  1.  1.]
     [nan -1. -1. -1. -1.]]

    >>> X4 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
    >>> dt_axis0 = DifferenceTransformer(axis=0)
    >>> Xt4 = dt_axis0.fit_transform(X4, axis=0)
    >>> print(Xt4)
    [[nan nan]
     [ 1. -1.]
     [ 1. -1.]
     [ 1. -1.]
     [ 1. -1.]]
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(self, order=1, axis=1):
        if not isinstance(order, int) or order < 1:
            raise ValueError(f"`order` must be a positive integer, but got {order}")
        self.order = order
        super().__init__(axis=axis)

    def _transform(self, X, y=None):
        """
        Perform the n-th order differencing transformation.

        Parameters
        ----------
        X : np.ndarray

        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : np.ndarray
            Transformed version of X with the same shape, containing the
            n-th order difference.
            The first `order` elements along the time axis are NaN.
        """
        diff_X = np.diff(X, n=self.order, axis=self.axis)

        # Check if diff_X is integer type.
        # If so, cast to float to allow inserting np.nan.
        if not np.issubdtype(diff_X.dtype, np.floating):
            diff_X = diff_X.astype(np.float64)

        # Insert the NaN at the beginning
        nan_shape = list(X.shape)
        nan_shape[self.axis] = self.order
        nans_to_prepend = np.full(nan_shape, np.nan, dtype=np.float64)

        Xt = np.concatenate([nans_to_prepend, diff_X], axis=self.axis)

        return Xt
