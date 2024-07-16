"""Wrapper for sklearn StandardScaler."""

__maintainer__ = []
__all__ = ["TimeSeriesScaler"]

import numpy as np
from sklearn.preprocessing import StandardScaler

from aeon.transformations.collection.base import BaseCollectionTransformer


class TimeSeriesScaler(BaseCollectionTransformer):
    """StandardScaler for time series.

    This class wraps the sklearn StandardScaler so that is standardises time series
    rather than ove rtime points. We need this because we store collections of time
    series in arrays of shape (n_cases, n_channels, n_timepoints). Standard scaler would
    transform a single series (n_channels, n_timepoints) so that each time point is zero
    mean, unit std dev. We want each channel to be zero mean, unit std dev. This is
    easily achieved by transposing, but its easy to get wrong and hard to detect if done
    wrong, hence the wrapper.

    The standard score of a time series `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the time series or zero if `with_mean=False`,
    and `s` is the standard deviation of the time series or one if
    `with_std=False`.

    Centering and scaling happen independently on each time series.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    See Also
    --------
    sklearn.preprocessing.StandardScaler : class wrapped by this class.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform. We have note used the copy parameter for in place copying, because it
    doesnt seem to work.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    Examples
    --------
    >>> from aeon.transformations.collection import TimeSeriesScaler
    >>> series = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]])
    >>> scaler = TimeSeriesScaler()
    >>> print(scaler.fit(series))
    TimeSeriesScaler()
    >>> scaler.transform(series)[0][0]
    array([0., 0., 0.])
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X into the catch22 features.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape ``(n_cases, n_channels, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
                of shape ``[n_cases]``, 2D np.ndarray `
                `(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is length
                of series i.

        Returns
        -------
        Xt : same shape as input
        """
        X2 = [None] * len(X)
        for i in range(len(X)):
            x1 = np.transpose(X[i])
            norm = self.scaler.fit_transform(x1)
            x1 = np.transpose(norm)
            X2[i] = x1
        if isinstance(X, np.ndarray):
            X2 = np.array(X2)
        return X2
