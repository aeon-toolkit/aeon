"""Normalization like z-normalization, standardization and min-max scaling."""

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Normalise(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies z-normalization
    applied along the timepoints axis (the last axis).
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the seleted normalisation.

        keepdims=True: bool, Retains the reduced axes with size one in the output,
        preserving the number of dimensions of the array.

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
        X_transformed : np.ndarray or list
            The normalized data.

        Example
        -------
        >>> from aeon.transformations.collection import Normalise
        >>> from aeon.datasets import load_unit_test
        >>> X, y = load_unit_test()

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


class MinMax(BaseCollectionTransformer):
    """MinMax transformer for collections.

    This transformer scales a collection of time series data to a specified range
    [min,max] along the time axis. The default is to [0,1]. For multivariate,
    it scales each channel independently.


    Parameters
    ----------
    min: float, default=0
        Minumum value of the range to scale to.
    max: float, default=1
        Maximum value of the range to scale to.
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
    }

    def __init__(self, min: float = 0, max: float = 1):
        self.min = min
        self.max = max
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the MinMax normalisation.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform.
        y: None
           Ignored.

        Returns
        -------
        X_transformed : np.ndarray or list
            The data transformed onto [min,max] scale.
        """
        if self.min > self.max:
            raise ValueError(
                f"min value {self.min} should be less than max value {self.max}"
            )

        if isinstance(X, np.ndarray):
            # Case 1: Equal length, X is a single 3D array
            min_val = np.min(X, axis=-1, keepdims=True)
            max_val = np.max(X, axis=-1, keepdims=True)
            range_val = np.where(
                max_val - min_val == 0, 1, max_val - min_val
            )  # Prevent division by zero
            X_scaled = (X - min_val) / range_val
            X_scaled = self.min + X_scaled * (self.max - self.min)
            return X_scaled
        else:
            # Case 2: X is a list of 2D arrays
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
