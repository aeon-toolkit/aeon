"""Wrapper for sklearn StandardScaler."""

__author__ = ["TonyBagnall", "dguijo"]
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
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array, a copy may still be returned.
    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently, unit standard
        deviation).
    inverse_transform_needed : bool, default=False
        If True, the inverse_transform method will be available.

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
    }

    def __init__(
        self, copy=True, with_mean=True, with_std=True, inverse_transform_needed=False
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.inverse_transform_needed = inverse_transform_needed
        if self.inverse_transform_needed:
            self._tags["capability:inverse_transform"] = True
        super().__init__()

    def _transform(self, X, y=None):
        """Scale X using StandardScaler.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            3D np.ndarray (any number of channels, equal length series) of shape
            ``(n_instances, n_channels, n_timepoints)`` or list of numpy arrays (any
            number of channels, unequal length series) of shape ``[n_instances]``, 2D
            np.ndarray ``(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is
            length of series i.

        Returns
        -------
        X_scaled : np.array
            Scaled time series dataset.
        """
        X_scaled = [None] * len(X)
        if self.inverse_transform_needed:
            self.individual_scalers_ = [None] * len(X)
        for i in range(len(X)):
            self.scaler = StandardScaler(
                copy=self.copy, with_mean=self.with_mean, with_std=self.with_std
            )
            x1 = np.transpose(X[i])
            norm = self.scaler.fit_transform(x1)
            X_scaled[i] = np.transpose(norm)

            if self.inverse_transform_needed:
                self.individual_scalers_[i] = self.scaler

        if isinstance(X, np.ndarray):
            X_scaled = np.array(X_scaled)
        return X_scaled

    def _inverse_transform(self, X, y=None):
        """Perform the inverse transform over X using fitted StandardScaler.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            3D np.ndarray (any number of channels, equal length series) of shape
            ``(n_instances, n_channels, n_timepoints)`` or list of numpy arrays (any
            number of channels, unequal length series) of shape ``[n_instances]``, 2D
            np.ndarray ``(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is
            length of series i.

        Returns
        -------
        X_original : np.array
            Original time series dataset.
        """
        X_original = [None] * len(X)
        for i in range(len(X)):
            x1 = np.transpose(X[i])
            norm = self.individual_scalers_[i].inverse_transform(x1)
            X_original[i] = np.transpose(norm)
        if isinstance(X, np.ndarray):
            X_original = np.array(X_original)
        return X_original
