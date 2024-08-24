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

from typing import Optional

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


class Normalise(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    Parameters
    ----------
    method : str, optional (default="z_norm")
        The normalization method to apply.
        Supported methods: "z_norm", "standardize", "min_max".
    axis : int, optional (default=2)
        Axis along which to apply the normalization.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(self, method: str = "z-norm", axis: int = 2):
        self.method = method
        self.axis = axis
        super().__init__()

    def _transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform method to apply the seleted normalisation.

        keepdims=True: bool, Retains the reduced axes with size one in the output,
        preserving the number of dimensions of the array.

        Parameters
        ----------
        X: np.ndarray
            The input samples to transform.
        y: array-like or list, optional
            The class values for X (not used in this method).

        Returns
        -------
        X_transformed : np.ndarray
            The normalized data.
        """
        if self.method in {"z_norm", "standardize"}:
            mean = np.mean(X, axis=self.axis, keepdims=True)
            std = np.std(X, axis=self.axis, keepdims=True)

            # Handle cases where std is 0
            std_nonzero = np.where(std == 0, 1, std)
            return (X - mean) / std_nonzero

        elif self.method == "min_max":
            min_val = np.min(X, axis=self.axis, keepdims=True)
            max_val = np.max(X, axis=self.axis, keepdims=True)

            # Prevent division by zero in case min_val == max_val
            range_nonzero = np.where(max_val == min_val, 1, max_val - min_val)
            return (X - min_val) / range_nonzero

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
