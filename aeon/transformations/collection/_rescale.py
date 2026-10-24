"""Rescaler classes for z-normalization, centering and min-max scaling."""

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Normalizer(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies z-normalization  applied along the timepoints axis (the
    last axis). For multivariate data, it normalizes each channel independently.

    Parameters
    ----------
    mean: float, default=0.0
        Mean value to scale to.
    std: float, default=1.0
        Standard deviation value to scale to.
    axis: int, default=-1
        Axis along which to compute the mean and standard deviation. Default is -1,
        which corresponds to the last axis (timepoints axis).

    When working with unequal length series, Normalizer is only able to
    use list of 2D arrays with shape (n_channels, n_timepoints_i).

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
            mean_val = np.mean(X, axis=self.axis, keepdims=True)
            std_val = np.std(X, axis=self.axis, keepdims=True)
            std_val = np.where(std_val == 0, 1, std_val)  # Prevent division by zero
            X_standardized = (X - mean_val) / std_val * self.std + self.mean
            return X_standardized

        else:
            # Case 2: X is a list of 2D arrays
            Xt = []
            for x in X:
                mean_val = np.mean(x, axis=-1, keepdims=True)
                std_val = np.std(x, axis=-1, keepdims=True)
                std_val = np.where(std_val == 0, 1, std_val)  # Prevent division by zero
                x_standardized = (x - mean_val) / std_val * self.std + self.mean
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
    axis: int, default=-1
        Axis along which to compute the min and max. Default is -1, which corresponds to
        the last axis (timepoints axis).

    When working with unequal length series, MinMaxScaler is only able to
    use list of 2D arrays with shape (n_channels, n_timepoints_i).

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
            min_val = np.min(X, axis=self.axis, keepdims=True)
            max_val = np.max(X, axis=self.axis, keepdims=True)
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

    Parameters
    ----------
    mean: float, default=0.0
        Mean value to scale to.
    axis: int, default=-1
        Axis along which to compute the mean. Default is -1, which corresponds to the
        last axis (timepoints axis).

    When working with unequal length series, Centerer is only able to
    use list of 2D arrays with shape (n_channels, n_timepoints_i).

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
            mean_val = np.mean(X, axis=self.axis, keepdims=True)
            Xt = X - mean_val + self.mean
            return Xt

        else:
            # Case 2: X is a list of 2D arrays
            Xt = []
            for x in X:
                mean_val = np.mean(x, axis=-1, keepdims=True)
                xt = x - mean_val + self.mean
                Xt.append(xt)
            return Xt
