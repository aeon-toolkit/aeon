"""Time Series imputer."""

__maintainer__ = []
__all__ = ["SimpleImputer"]

from typing import Callable, Optional, Union

import numpy as np
from scipy.stats import mode

from aeon.transformations.collection.base import BaseCollectionTransformer


class SimpleImputer(BaseCollectionTransformer):
    """Time series imputer.

    Transformer that imputes missing values in time series. Fill values are calculated
    across series.

    Parameters
    ----------
    strategy : str or Callable, default="mean"
        The imputation strategy.
            - if "mean", replace missing values using the mean.
            - if "median", replace missing values using the median.
            - if "constant", replace missing values with the fill_value.
            - if "most frequent", replace missing values with the most frequent value.
            - if Callable, a function that returns the value to replace
            missing values with on each 1D array containing all
            non-missing values of each series.

    fill_value : float or None, default=None
        The value to replace missing values with. Only used when strategy is "constant".
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "removes_missing_values": True,
    }

    def __init__(
        self,
        strategy: Union[str, Callable] = "mean",
        fill_value: Optional[float] = None,
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
        np.ndarray or list
        """
        self._validate_parameters()

        if isinstance(X, np.ndarray):  # if X is a 3D array

            if self.strategy == "mean":
                X = np.where(np.isnan(X), np.nanmean(X, axis=-1, keepdims=True), X)

            elif self.strategy == "median":
                X = np.where(np.isnan(X), np.nanmedian(X, axis=-1, keepdims=True), X)

            elif self.strategy == "constant":
                X = np.where(np.isnan(X), self.fill_value, X)

            elif self.strategy == "most frequent":
                X = np.where(
                    np.isnan(X),
                    mode(X, axis=-1, nan_policy="omit", keepdims=True).mode,
                    X,
                )

            else:  # if strategy is a callable function
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        nan_mask = np.isnan(X[i, j])
                        X[i, j] = np.where(
                            nan_mask, self.strategy(X[i, j][nan_mask]), X[i, j]
                        )  # applying callable function to each case without nan values
            return X

        else:  # if X is a list of 2D arrays
            Xt = []

            for x in X:
                if self.strategy == "mean":
                    x = np.where(np.isnan(x), np.nanmean(x, axis=-1, keepdims=True), x)
                elif self.strategy == "median":
                    x = np.where(
                        np.isnan(x), np.nanmedian(x, axis=-1, keepdims=True), x
                    )
                elif self.strategy == "constant":
                    x = np.where(np.isnan(x), self.fill_value, x)
                elif self.strategy == "most frequent":
                    x = np.where(
                        np.isnan(x),
                        mode(x, axis=-1, nan_policy="omit", keepdims=True).mode,
                        x,
                    )
                else:  # if strategy is a callable function
                    n_channels = x.shape[0]
                    for i in range(n_channels):
                        nan_mask = np.isnan(x[i])
                        x[i] = np.where(nan_mask, self.strategy(x[i][nan_mask]), x[i])
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
