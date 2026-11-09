"""Rescaler classes for z-normalization, centering and min-max scaling."""

import numpy as np

from aeon.transformations.collection.base import (
    BaseCollectionTransformer,
    CollectionInverseTransformerMixin,
)


class Normalizer(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies z-normalization applied along the timepoints axis (the
    last axis). For multivariate data, it normalizes each channel independently.

    Examples
    --------
    >>> from aeon.transformations.collection import Normalizer
    >>> import numpy as np
    >>> X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> normaliser = Normalizer()
    >>> Xt = normaliser.fit_transform(X)
    >>> mean=np.mean(Xt, axis=-1)
    >>> std = np.std(Xt, axis=-1)
    >>> assert np.allclose(mean, 0)
    >>> assert np.allclose(std, 1)
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the normalisation.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays of shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray or list
            The normalized data.
        """
        if isinstance(X, np.ndarray):
            # Case 1: X is a single 3D array
            mean_val = np.mean(X, axis=-1, keepdims=True)
            std_val = np.std(X, axis=-1, keepdims=True)
            std_val = np.where(std_val == 0, 1, std_val)  # Prevent division by zero
            X_standardized = (X - mean_val) / std_val
            return X_standardized

        else:
            # Case 2: X is a list of 2D arrays
            Xt = []
            for x in X:
                mean_val = np.mean(x, axis=-1, keepdims=True)
                std_val = np.std(x, axis=-1, keepdims=True)
                std_val = np.where(std_val == 0, 1, std_val)  # Prevent division by zero
                x_standardized = (x - mean_val) / std_val
                Xt.append(x_standardized)
            return Xt


class MinMaxScaler(BaseCollectionTransformer):
    """MinMax transformer for collections.

    This transformer scales a collection of time series data to a specified range
    [min,max] along the time axis. The default is to [0,1]. For multivariate,
    it scales each channel independently.


    Parameters
    ----------
    min: float, default=0
        Minimum value of the range to scale to.
    max: float, default=1
        Maximum value of the range to scale to.

    Examples
    --------
    >>> from aeon.transformations.collection import MinMaxScaler
    >>> import numpy as np
    >>> X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> minmax = MinMaxScaler()
    >>> Xt = minmax.fit_transform(X)
    >>> min_val = np.min(Xt, axis=-1)
    >>> max_val = np.max(Xt, axis=-1)
    >>> assert np.allclose(min_val, 0)
    >>> assert np.allclose(max_val, 1)

    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, min: float = 0, max: float = 1):
        self.min = min
        self.max = max
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the MinMaxScaler normalisation.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either a list of 2D arrays of shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y: None
           Ignored.

        Returns
        -------
        np.ndarray or list
            The data transformed onto [min,max] scale.
        """
        if self.min > self.max:
            raise ValueError(
                f"min value {self.min} should be less than max value {self.max}"
            )

        if isinstance(X, np.ndarray):
            # Case 1: Equal length series, X is a single 3D array
            min_val = np.min(X, axis=-1, keepdims=True)
            max_val = np.max(X, axis=-1, keepdims=True)
            range_val = np.where(
                max_val - min_val == 0, 1, max_val - min_val
            )  # Prevent division by zero
            X_scaled = (X - min_val) / range_val
            X_scaled = self.min + X_scaled * (self.max - self.min)
            return X_scaled
        else:
            # Case 2: Unequal length series, X is a list of 2D arrays
            Xt = []
            for x in X:
                min_val = np.min(x, axis=-1, keepdims=True)
                max_val = np.max(x, axis=-1, keepdims=True)
                range_val = np.where(
                    max_val - min_val == 0, 1, max_val - min_val
                )  # Prevent division by zero
                x_scaled = (x - min_val) / range_val
                x_scaled = self.min + x_scaled * (self.max - self.min)
                Xt.append(x_scaled)
            return Xt


class Centerer(BaseCollectionTransformer):
    """Centering transformer for collections.

    This transformer recentres series to have zero mean, but does not change the
    variance. For multivariate data, it normalizes each channel independently.

    Examples
    --------
    >>> from aeon.transformations.collection import Centerer
    >>> import numpy as np
    >>> X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> recentre = Centerer()
    >>> Xt = recentre.fit_transform(X)
    >>> mean=np.mean(Xt, axis=-1)
    >>> assert np.allclose(mean, 0)
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to center series.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays of shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray or list
            The centered data.
        """
        if isinstance(X, np.ndarray):
            # Case 1: X is a single 3D array
            mean_val = np.mean(X, axis=-1, keepdims=True)
            Xt = X - mean_val
            return Xt

        else:
            # Case 2: X is a list of 2D arrays
            Xt = []
            for x in X:
                mean_val = np.mean(x, axis=-1, keepdims=True)
                xt = x - mean_val
                Xt.append(xt)
            return Xt


class GlobalNormalizer(CollectionInverseTransformerMixin, BaseCollectionTransformer):
    """Global z-normalization transformer for time series collections.

    Z-normalizes each channel using the mean and standard deviation
    computed across all cases and timepoints during fit.

    Parameters
    ----------
    mean: float, default=0.0
        The mean value to which the data will be normalized.
    std: float, default=1.0
        The standard deviation value to which the data will be normalized.
    """

    _tags = {
        "X_inner_type": ["numpy2D", "numpy3D", "np-list"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:inverse_transform": True,
    }

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def _fit(self, X, y=None):
        """
        Fit method to compute per-channel means and stds for the z-normalization.

        Parameters
        ----------
        X : np.ndarray or list
        y : None
            Ignored.
        """
        if isinstance(X, list):
            X = np.concatenate(X, axis=1)
            ax = 1
        elif X.ndim == 2:
            ax = (0, 1)
        elif X.ndim == 3:
            ax = (0, 2)

        self._mean = np.nanmean(X, axis=ax)
        self._std = np.nanstd(X, axis=ax)
        self._std = np.where(self._std == 0, 1, self._std)

    def _transform(self, X, y=None):
        """
        Transform method to apply the z-normalization.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either:
            - a 3D array of shape ``(n_cases, n_channels, n_timepoints)``
            - a list of ``n_cases`` 2D arrays of shape ``(n_channels, n_timepoints_i)``
            - a univariate 2D array of shape ``(n_cases, n_timepoints)``.
        y : None
            Ignored.
        """
        if isinstance(X, list):
            return [self._transform(x_i) for x_i in X]
        if X.ndim == 2:
            m = self._mean.reshape((-1, 1))
            s = self._std.reshape((-1, 1))
        elif X.ndim == 3:
            m = self._mean.reshape((1, -1, 1))
            s = self._std.reshape((1, -1, 1))
        return (X - m) / s * self.std + self.mean

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform method to revert the z-normalization.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to inverse transform. Either:
            - a 3D array of shape ``(n_cases, n_channels, n_timepoints)``
            - a list of ``n_cases`` 2D arrays of shape ``(n_channels, n_timepoints_i)``
            - a univariate 2D array of shape ``(n_cases, n_timepoints)``.
        y : Ignored.
        """
        if isinstance(X, list):
            return [self._inverse_transform(x_i) for x_i in X]
        if X.ndim == 2:
            m = self._mean.reshape((-1, 1))
            s = self._std.reshape((-1, 1))
        elif X.ndim == 3:
            m = self._mean.reshape((1, -1, 1))
            s = self._std.reshape((1, -1, 1))
        return (X - self.mean) / self.std * s + m


class GlobalMinMaxScaler(CollectionInverseTransformerMixin, BaseCollectionTransformer):
    """Global MinMaxScaler transformer for collections.

    Normalizes collection bounds for each channel using the minimum and maximum
    computed across all cases and timepoints during fit.

    Parameters
    ----------
    min: float, default=0.0
        The minimum value of the range to scale to.
    max: float, default=1.0
        The maximum value of the range to scale to.
    """

    _tags = {
        "X_inner_type": ["numpy2D", "numpy3D", "np-list"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:inverse_transform": True,
    }

    def __init__(self, min: float = 0.0, max: float = 1.0):
        super().__init__()

        self.min = min
        self.max = max

    def _fit(self, X, y=None):
        """
        Fit method to compute per-channel mins and maxs for the min-max scaling.

        Parameters
        ----------
        X : np.ndarray or list
        y :  None
            Ignored.
        """
        if self.min >= self.max:
            raise ValueError(
                f"min value {self.min} should be less than max value {self.max}"
            )
        if isinstance(X, list):
            X = np.concatenate(X, axis=1)
            ax = 1
        elif X.ndim == 2:
            ax = (0, 1)
        elif X.ndim == 3:
            ax = (0, 2)

        self._min = np.nanmin(X, axis=ax)
        self._max = np.nanmax(X, axis=ax)
        # prevent division by zero in case of constant series
        delta = self._max - self._min
        self._max = np.where(delta == 0, self._min + 1, self._max)

    def _transform(self, X, y=None):
        """
        Transform method to apply the min-max scaling.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either:
            - a 3D array of shape ``(n_cases, n_channels, n_timepoints)``
            - a list of ``n_cases`` 2D arrays of shape ``(n_channels, n_timepoints_i)``
            - a univariate 2D array of shape ``(n_cases, n_timepoints)``.
        y : Ignored.
        """
        if isinstance(X, list):
            return [self._transform(x_i) for x_i in X]
        if X.ndim == 2:
            m = self._min.reshape((-1, 1))
            M = self._max.reshape((-1, 1))
        elif X.ndim == 3:
            m = self._min.reshape((1, -1, 1))
            M = self._max.reshape((1, -1, 1))
        return (X - m) / (M - m) * (self.max - self.min) + self.min

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform method to revert the min-max scaling.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to inverse transform. Either:
            - a 3D array of shape ``(n_cases, n_channels, n_timepoints)``
            - a list of ``n_cases`` 2D arrays of shape ``(n_channels, n_timepoints_i)``
            - a univariate 2D array of shape ``(n_cases, n_timepoints)``.
        y : Ignored.
        """
        if isinstance(X, list):
            return [self._inverse_transform(x_i) for x_i in X]
        if X.ndim == 2:
            m = self._min.reshape((-1, 1))
            M = self._max.reshape((-1, 1))
        elif X.ndim == 3:
            m = self._min.reshape((1, -1, 1))
            M = self._max.reshape((1, -1, 1))
        X = (X - self.min) / (self.max - self.min)
        return X * (M - m) + m


class GlobalCenterer(CollectionInverseTransformerMixin, BaseCollectionTransformer):
    """Global Centerer transformer for collections.

    Centers each channel using the mean computed across all cases and timepoints during
    fit.

    Parameters
    ----------
    mean: float, default=0.0
        The mean value to which the data will be recentered.
    """

    _tags = {
        "X_inner_type": ["numpy2D", "numpy3D", "np-list"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:inverse_transform": True,
    }

    def __init__(self, mean=0.0):
        super().__init__()
        self.mean = mean

    def _fit(self, X, y=None):
        """
        Fit method to compute per-channel means for centering.

        Parameters
        ----------
        X : np.ndarray or list
        y :  None
            Ignored.
        """
        if isinstance(X, list):
            X = np.concatenate(X, axis=1)
            ax = 1
        elif X.ndim == 2:
            ax = (0, 1)
        elif X.ndim == 3:
            ax = (0, 2)

        self._mean = np.nanmean(X, axis=ax, keepdims=True)

    def _transform(self, X, y=None):
        """
        Transform method to apply centering.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either:
            - a 3D array of shape ``(n_cases, n_channels, n_timepoints)``
            - a list of ``n_cases`` 2D arrays of shape ``(n_channels, n_timepoints_i)``
            - a univariate 2D array of shape ``(n_cases, n_timepoints)``.
        y : Ignored.
        """
        if isinstance(X, list):
            return [self._transform(x_i) for x_i in X]
        if X.ndim == 2:
            m = self._mean.reshape((-1, 1))
        elif X.ndim == 3:
            m = self._mean.reshape((1, -1, 1))
        return X - m + self.mean

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform method to revert centering.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to inverse transform. Either:
            - a 3D array of shape ``(n_cases, n_channels, n_timepoints)``
            - a list of ``n_cases`` 2D arrays of shape ``(n_channels, n_timepoints_i)``
            - a univariate 2D array of shape ``(n_cases, n_timepoints)``.
        y : Ignored.
        """
        if isinstance(X, list):
            return [self._inverse_transform(x_i) for x_i in X]
        if X.ndim == 2:
            m = self._mean.reshape((-1, 1))
        elif X.ndim == 3:
            m = self._mean.reshape((1, -1, 1))
        return X + m - self.mean
