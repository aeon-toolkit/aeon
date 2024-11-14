"""Time Series imputer."""

__maintainer__ = []
__all__ = ["SimpleImputer"]

from collections import Counter
from typing import Callable, Union

import numpy as np
from scipy.stats import mode

from aeon.transformations.collection.base import BaseCollectionTransformer


class SimpleImputer(BaseCollectionTransformer):
    """Time series imputer.

    Transformer that imputes missing values in time series. It is fitted on each channel
    independently. After transformation the collection will be a 3D numpy array shape
    (n_cases, n_channels, length).

    Parameters
    ----------
    strategy : str or Callable, default="mean"
        The imputation strategy.
            - if "mean", replace missing values using the mean.
            - if "median", replace missing values using the median.
            - if "constant", replace missing values with the fill_value,
            works with str or int.
            - if "most frequent", replace missing values with the most frequent value,
            works with str or int.
            - if Callable, a function that returns the value to replace
            missing values with on each 1D array containing all
            non-missing values of each series.

    fill_value : float, int, str or None, default=None
        The value to replace missing values with. Only used when strategy is "constant".
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "removes_missing_values": True,
    }

    def __init__(
        self,
        strategy: Union[str, Callable] = "mean",
        fill_value: Union[int, float, str, None] = None,
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        super().__init__()

    def _transform(
        self, X: Union[np.ndarray, list[np.ndarray]], y=None
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Transform method to apply the SimpleImputer.

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
        np.ndarray
        """
        self._validate_parameters()

        if isinstance(X, np.ndarray):  # if X is a 3D array

            if self.strategy == "mean":
                X = np.where(np.isnan(X), np.nanmean(X, axis=(0, 2), keepdims=True), X)

            elif self.strategy == "median":
                X = np.where(
                    np.isnan(X), np.nanmedian(X, axis=(0, 2), keepdims=True), X
                )

            elif self.strategy == "constant":
                if np.issubdtype(X.dtype, np.str_):  # if X is a string array
                    fill_values = np.array([self.fill_value])
                    fill_values = np.broadcast_to(fill_values, X.shape)
                    X[X == "nan"] = fill_values[X == "nan"]
                else:
                    X = np.where(np.isnan(X), self.fill_value, X)

            elif self.strategy == "most frequent":
                if np.issubdtype(X.dtype, np.str_):  # if X is a string array
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            X[i, j][X[i, j] == "nan"] = Counter(X[i, j]).most_common(1)[
                                0
                            ][0]
                else:
                    modes = [
                        mode(X[:, i, :].flatten(), nan_policy="omit").mode
                        for i in range(X.shape[1])
                    ]
                    modes = np.array(modes).reshape(1, X.shape[1], 1)
                    X = np.where(
                        np.isnan(X),
                        modes,
                        X,
                    )

            else:  # if strategy is a callable function
                fill_values = []
                for i in range(X.shape[1]):
                    non_nan_1d_array = X[:, i, :][~np.isnan(X[:, i, :])].flatten()
                    fill_values.append(self.strategy(non_nan_1d_array))
                fill_values = np.array(fill_values).reshape(1, X.shape[1], 1)
                X = np.where(np.isnan(X), fill_values, X)
            return X

        else:  # if X is a list of 2D arrays
            channels = X[0].shape[0]

            non_missing_values = [[] for _ in range(channels)]
            for x in X:
                for i in range(channels):
                    non_missing_values[i].extend(x[i][~np.isnan(x[i])])
            non_missing_values = np.array([np.array(x) for x in non_missing_values])

            if self.strategy == "mean":
                fill_values = np.mean(non_missing_values, axis=1, keepdims=True)
            elif self.strategy == "median":
                fill_values = np.median(non_missing_values, axis=1, keepdims=True)
            elif self.strategy == "constant":
                fill_values = self.fill_value
            elif self.strategy == "most frequent":
                if np.issubdtype(non_missing_values[0].dtype, np.str_):
                    fill_values = [
                        Counter(non_missing_values[i]).most_common(1)[0][0]
                        for i in range(channels)
                    ]
                else:
                    fill_values = mode(non_missing_values, axis=1, keepdims=True).mode
            else:
                fill_values = [
                    self.strategy(non_missing_values[i]) for i in range(channels)
                ]
                fill_values = np.array(fill_values).reshape(channels, 1)

            Xt = []
            for x in X:
                if np.issubdtype(x.dtype, np.str_):
                    nan_mask = x == "nan"
                    if self.strategy == "constant":
                        x[nan_mask] = fill_values
                    else:
                        x[nan_mask] = np.broadcast_to(fill_values, x.shape)[nan_mask]
                else:
                    x = np.where(np.isnan(x), fill_values, x)
                Xt.append(x)
            return Xt

    def _validate_parameters(self):
        """Validate the parameters."""
        if self.strategy not in [
            "mean",
            "median",
            "constant",
            "most frequent",
        ] and not callable(self.strategy):
            raise ValueError(
                "strategy must be 'mean', 'median', 'constant', 'most frequent',"
                f" or a callable. Got {self.strategy}."
            )

        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'constant'.")

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"strategy": "mean"}
