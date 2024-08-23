"""Normalization techniques like z-normalization, standardization, and min-max scaling.

Classes
-------
Normalise : A transformer class for normalizing collections of time series data.
BaseCollectionTransformer : A base class template for all collection transformers.

The Normalise class supports several normalization methods that can be applied to
time series data along a specified axis. It extends the BaseCollectionTransformer
class and provides functionality to fit and transform data, ensuring consistent
scaling across datasets.
"""

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


class Normalise(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    Parameters
    ----------
    X: np.ndarray shape (n_cases, n_channels, n_timepoints)
        The training input samples.
    y: array-like or list, default=None
        The class values for X. If not specified, a random sample (i.e. not of the
        same class) will be used when computing the threshold for the Shapelet
        Occurrence feature.
    method : str, optional (default="z_norm")
        The normalization method to apply.
        The normalization method to apply.
        Supported methods: "z_norm", "standardize", "min_max".
    axis : int, optional (default=2)
        Axis along which to apply the normalization.
    """

    def __init__(self, method="z-norm", axis=2):
        self.method = method
        self.axis = axis
        super().__init__()

    def _fit(self, X, y=None):
        # Fit method has no operaations on class normalise
        pass

    """keepdims=True: Retains the reduced axes with size one in the output,
       preserving the number of dimensions of the array."""

    def _transform(self, X, y=None):
        if self.method == "z_norm":
            mean = np.mean(X, axis=self.axis, keepdims=True)
            std = np.std(X, axis=self.axis, keepdims=True)
            return (X - mean) / std
        elif self.method == "standardize":
            mean = np.mean(X, axis=self.axis, keepdims=True)
            std = np.std(X, axis=self.axis, keepdims=True)
            return (X - mean) / std
        elif self.method == "min_max":
            min_val = np.min(X, axis=self.axis, keepdims=True)
            max_val = np.max(X, axis=self.axis, keepdims=True)
            return (X - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

            raise ValueError(f"Unknown normalization method: {self.method}")
