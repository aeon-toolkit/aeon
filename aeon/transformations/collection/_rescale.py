"""Rescaler classes for z-normalization, centering and min-max scaling."""

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Normalizer(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies z-normalization applied along the specified timepoints axis
    (by default, the last axis). For multivariate data, normalization is performed
    independently for each channel. When working with unequal length series axis,
    is relative to the numpy arrays dimensions and could be either 0 or 1.

    Parameters
    ----------
    mean: float, default=0.0
        Mean value to scale to.
    std: float, default=1.0
        Standard deviation value to scale to.
    axis: int, default=-1
        Axis along which to compute the mean and standard deviation. Default is -1,
        which corresponds to the last axis (timepoints axis).

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
        "X_inner_type": ["numpy2D", "numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, mean: float = 0.0, std: float = 1.0, axis: int = -1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.axis = axis

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the normalisation.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either:
            - a 3D array of shape :
                + if axis = -1: ``(n_cases, n_channels, n_timepoints)``
                + if axis = 1: ``(n_cases, n_timepoints, n_channels)``
            - a list of ``n_cases`` 2D arrays
                + if axis = -1: ``(n_channels, n_timepoints_i)``
                + if axis = 0: ``(n_timepoints_i, n_channels)``
            - a 2D array of shape :
                + if axis = -1: ``(n_channels, n_timepoints)``
                    or ``(n_cases, n_timepoints)``
                + if axis = 0: ``(n_timepoints, n_channels)``
        y : None
            Ignored.

        Returns
        -------
        np.ndarray or list
            The normalized data.
        """
        if isinstance(X, np.ndarray):
            # Case 1: X is a single 3D array
            mean_val = np.nanmean(X, axis=self.axis, keepdims=True)
            std_val = np.nanstd(X, axis=self.axis, keepdims=True)
            std_val = np.where(std_val == 0, 1, std_val)  # Prevent division by zero
            return (X - mean_val) / std_val * self.std + self.mean
        else:
            # Case 2: X is a list of 2D arrays
            return [self._transform(x) for x in X]


class MinMaxScaler(BaseCollectionTransformer):
    """MinMax transformer for collections.

    This transformer scales a collection of time series data to a specified range
    [min,max] along the time axis. The default is to [0,1]. For multivariate,
    it scales each channel independently. When working with unequal length series, axis
    is relative to the numpy arrays dimensions and could be either 0 or 1.


    Parameters
    ----------
    min: float, default=0
        Minimum value of the range to scale to.
    max: float, default=1
        Maximum value of the range to scale to.
    axis: int, default=-1
        Axis along which to compute the min and max. Default is -1, which corresponds to
        the last axis (timepoints axis).

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
        "X_inner_type": ["numpy2D", "numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, min: float = 0, max: float = 1, axis: int = -1):
        super().__init__()
        self.min = min
        self.max = max
        self.axis = axis

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the MinMaxScaler normalisation.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either:
            - a 3D array of shape :
                + if axis = -1: ``(n_cases, n_channels, n_timepoints)``
                + if axis = 1: ``(n_cases, n_timepoints, n_channels)``
            - a list of ``n_cases`` 2D arrays
                + if axis = -1: ``(n_channels, n_timepoints_i)``
                + if axis = 0: ``(n_timepoints_i, n_channels)``
            - a 2D array of shape :
                + if axis = -1: ``(n_channels, n_timepoints)``
                    or ``(n_cases, n_timepoints)``
                + if axis = 0: ``(n_timepoints, n_channels)``
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
            min_val = np.nanmin(X, axis=self.axis, keepdims=True)
            max_val = np.nanmax(X, axis=self.axis, keepdims=True)
            # Prevent division by zero
            delta = max_val - min_val
            delta = np.where(delta == 0, 1, delta)
            return (X - min_val) / delta * (self.max - self.min) + self.min
        else:
            # Case 2: Unequal length series, X is a list of 2D arrays
            return [self._transform(x) for x in X]


class Centerer(BaseCollectionTransformer):
    """Centering transformer for collections.

    This transformer recentres series to have zero mean, but does not change the
    variance. For multivariate data, it normalizes each channel independently.
    When working with unequal length series, axis is relative to the numpy arrays
    dimensions and could be either 0 or 1.

    Parameters
    ----------
    mean: float, default=0.0
        Mean value to scale to.
    axis: int, default=-1
        Axis along which to compute the mean. Default is -1, which corresponds to the
        last axis (timepoints axis).

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
        "X_inner_type": ["numpy2D", "numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, mean: float = 0.0, axis: int = -1):
        self.mean = mean
        self.axis = axis
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to center series.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either:
            - a 3D array of shape :
                + if axis = -1: ``(n_cases, n_channels, n_timepoints)``
                + if axis = 1: ``(n_cases, n_timepoints, n_channels)``
            - a list of ``n_cases`` 2D arrays
                + if axis = -1: ``(n_channels, n_timepoints_i)``
                + if axis = 0: ``(n_timepoints_i, n_channels)``
            - a 2D array of shape :
                + if axis = -1: ``(n_channels, n_timepoints)``
                    or ``(n_cases, n_timepoints)``
                + if axis = 0: ``(n_timepoints, n_channels)``
        y : None
            Ignored.

        Returns
        -------
        np.ndarray or list
            The centered data.
        """
        if isinstance(X, np.ndarray):
            # Case 1: X is a single 3D array
            mean_val = np.nanmean(X, axis=self.axis, keepdims=True)
            return X - mean_val + self.mean

        else:
            return [self._transform(x) for x in X]
