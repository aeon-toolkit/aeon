"""Normalization like z-normalization, standardization and min-max scaling."""

from typing import Optional

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Normalise(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies different normalization techniques to time series data,
    ensuring that the data is scaled consistently across all samples. It supports
    methods such as z-normalization, standardization, and min-max scaling, which are
    applied along the timepoints axis (the last axis).

    This transformer converts all input data to 3D numpy arrays of shape
    (n_cases, n_channels, n_timepoints). Normalization is always applied along the
    timepoints axis. For input types that are not already 3D, the transformer handles
    the conversion internally.

    Parameters
    ----------
    method : str, optional (default="z_norm")
        The normalization method to apply.
        Supported methods: "z_norm", "standardize", "min_max".

        z_norm: Subtracts the mean and divides by the standard deviation
        along the specified axis. Used to center the data and standardize its variance,
        making it dimensionless. This is useful when comparing datasets with different
        units.

        standardize: Subtracts the mean along the specified axis. Used to center data.

        min_max: Useful when you need to normalize data to a bounded range, which is
        important for algorithms that require or perform better with inputs within a
        specific range.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(self, method: str = "z_norm"):
        self.method = method
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
        # Axis=-1 normalises across all the timepoints in the 3D array
        if self.method in {"z_norm"}:
            mean = np.mean(X, axis=-1, keepdims=True)
            std = np.std(X, axis=-1, keepdims=True)

            # Handle cases where std is 0
            std_nonzero = np.where(std == 0, 1, std)
            return (X - mean) / std_nonzero

        if self.method in {"standardize"}:
            mean = np.mean(X, axis=-1, keepdims=True)

            return X - mean

        elif self.method == "min_max":
            min_val = np.min(X, axis=-1, keepdims=True)
            max_val = np.max(X, axis=-1, keepdims=True)

            # Prevent division by zero in case min_val == max_val
            range_nonzero = np.where(max_val == min_val, 1, max_val - min_val)
            return (X - min_val) / range_nonzero

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
