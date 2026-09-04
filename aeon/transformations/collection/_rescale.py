"""Rescaler classes for z-normalization, centering and min-max scaling."""

import numpy as np

from aeon.transformations.collection.base import (
    BaseCollectionTransformer,
    BaseGlobalCollectionTransformer,
)


class Normalizer(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies z-normalization  applied along the timepoints axis (the
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
            Collection to transform. Either a list of 2D arrays with shape
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
            Collection to transform. Either a list of 2D arrays with shape
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
            Collection to transform. Either a list of 2D arrays with shape
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


class GlobalNormalizer(BaseGlobalCollectionTransformer):
    """Global Normaliser transformer for collections.

    This transformer applies z-normalization applied along the timepoints axis (the
    last axis). For multivariate data, it normalizes each channel independently.
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.x_means = None
        self.x_stds = None
        self.y_means = None
        self.y_stds = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Fit method to compute per-feature means and stds for the z-normalization.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to fit. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to fit. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        unequal_length = isinstance(X, list)
        if not unequal_length:
            self.x_means = np.mean(X, axis=(0, 2), keepdims=True)
            self.x_stds = np.std(X, axis=(0, 2), keepdims=True)
        else:
            self.x_means = np.mean(
                [np.mean(x, axis=-1, keepdims=True) for x in X], axis=0
            )
            self.x_stds = np.std([np.std(x, axis=-1, keepdims=True) for x in X], axis=0)

        if y is not None:
            unequal_length = isinstance(y, list)
            if not unequal_length:
                self.y_means = np.mean(y, axis=(0, 2), keepdims=True)
                self.y_stds = np.std(y, axis=(0, 2), keepdims=True)
            else:
                self.y_means = np.mean(
                    [np.mean(y_i, axis=-1, keepdims=True) for y_i in y], axis=0
                )
                self.y_stds = np.std(
                    [np.std(y_i, axis=-1, keepdims=True) for y_i in y], axis=0
                )

    def _f(self, x, means, stds):
        unequal_length = isinstance(x, list)
        if unequal_length:
            return [self._f(x_i, means, stds) for x_i in x]
        else:
            return (x - means) / stds * self.std + self.mean

    def _transform(self, X, y=None):
        """
        Transform method to apply the z-normalization.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        if y is None:
            return self._f(X, self.x_means, self.x_stds)
        else:
            return self._f(X, self.x_means, self.x_stds), self._f(
                y, self.y_means, self.y_stds
            )

    def _fi(self, x, means, stds):
        unequal_length = isinstance(x, list)
        if unequal_length:
            return [self._fi(x_i, means, stds) for x_i in x]
        else:
            return (x - self.mean) / self.std * stds + means

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform method to revert the z-normalization.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to inverse transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to inverse transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        if y is None:
            return self._fi(X, self.x_means, self.x_stds)
        else:
            return self._fi(X, self.x_means, self.x_stds), self._fi(
                y, self.y_means, self.y_stds
            )


class GlobalMinMaxScaler(BaseGlobalCollectionTransformer):
    """Global MinMax transformer for collections.

    This transformer scales a collection of time series data to a specified range
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, min: float = 0.0, max: float = 1.0):
        self.min = min
        self.max = max
        self.x_mins = None
        self.x_maxs = None
        self.y_mins = None
        self.y_maxs = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Fit method to compute per-feature mins and maxs for the min-max scaling.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to fit. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to fit. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        unequal_length = isinstance(X, list)
        if not unequal_length:
            self.x_mins = np.min(X, axis=(0, 2), keepdims=True)
            self.x_maxs = np.max(X, axis=(0, 2), keepdims=True)
        else:
            self.x_mins = np.min([np.min(x, axis=-1, keepdims=True) for x in X], axis=0)
            self.x_maxs = np.max([np.max(x, axis=-1, keepdims=True) for x in X], axis=0)

        if y is not None:
            unequal_length = isinstance(y, list)
            if not unequal_length:
                self.y_mins = np.min(y, axis=(0, 2), keepdims=True)
                self.y_maxs = np.max(y, axis=(0, 2), keepdims=True)
            else:
                self.y_mins = np.min(
                    [np.min(y_i, axis=-1, keepdims=True) for y_i in y], axis=0
                )
                self.y_maxs = np.max(
                    [np.max(y_i, axis=-1, keepdims=True) for y_i in y], axis=0
                )

    def _f(self, x, mins, maxs):
        unequal_length = isinstance(x, list)
        if unequal_length:
            return [self._f(x_i, mins, maxs) for x_i in x]
        else:
            x = (x - mins) / (maxs - mins)
            return x * (self.max - self.min) + self.min

    def _transform(self, X, y=None):
        """
        Transform method to apply the min-max scaling.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.

        y : np.ndarray or list (optional)
                    Collection to fit. Either a list of 2D arrays with shape
                    ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
                    ``(n_cases, n_channels, n_timepoints)``.
        """
        if y is None:
            return self._f(X, self.x_mins, self.x_maxs)
        else:
            return self._f(X, self.x_mins, self.x_maxs), self._f(
                y, self.y_mins, self.y_maxs
            )

    def _fi(self, x, mins, maxs):
        unequal_length = isinstance(x, list)
        if unequal_length:
            return [self._fi(x_i, mins, maxs) for x_i in x]
        else:
            x = (x - self.min) / (self.max - self.min)
            return x * (maxs - mins) + mins

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform method to revert the min-max scaling.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to inverse transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
                    Collection to fit. Either a list of 2D arrays with shape
                    ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
                    ``(n_cases, n_channels, n_timepoints)``.
        """
        if y is None:
            return self._fi(X, self.x_mins, self.x_maxs)
        else:
            return self._fi(X, self.x_mins, self.x_maxs), self._fi(
                y, self.y_mins, self.y_maxs
            )


class GlobalCenterer(BaseGlobalCollectionTransformer):
    """Global Centerer transformer for collections.

    This transformer centers a collection of time series data by subtracting the mean
    along the timepoints axis (the last axis). For multivariate data, it centers each
    channel independently.
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, mean=0.0):
        self.mean = mean
        self.x_means = None
        self.y_means = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Fit method to compute per-feature means for centering.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to fit. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to fit. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        unequal_length = isinstance(X, list)
        if not unequal_length:
            self.x_means = np.mean(X, axis=(0, 2), keepdims=True)
        else:
            self.x_means = np.mean(
                [np.mean(x, axis=-1, keepdims=True) for x in X], axis=0
            )

        if y is not None:
            unequal_length = isinstance(y, list)
            if not unequal_length:
                self.y_means = np.mean(y, axis=(0, 2), keepdims=True)
            else:
                self.y_means = np.mean(
                    [np.mean(y_i, axis=-1, keepdims=True) for y_i in y], axis=0
                )

    def _f(self, x, means):
        unequal_length = isinstance(x, list)
        if unequal_length:
            return [self._f(x_i, means) for x_i in x]
        else:
            return x - means + self.mean

    def _transform(self, X, y=None):
        """
        Transform method to apply centering.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        if y is None:
            return self._f(X, self.x_means)
        else:
            return self._f(X, self.x_means), self._f(y, self.y_means)

    def _fi(self, x, means):
        unequal_length = isinstance(x, list)
        if unequal_length:
            return [self._fi(x_i, means) for x_i in x]
        else:
            return x - self.mean + means

    def _inverse_transform(self, X, y=None):
        """
        Inverse transform method to revert centering.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to inverse transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : np.ndarray or list (optional)
            Collection to inverse transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        """
        if y is None:
            return self._fi(X, self.x_means)
        else:
            return self._fi(X, self.x_means), self._fi(y, self.y_means)
