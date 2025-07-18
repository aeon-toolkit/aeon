"""SETAR-Forest: An ensemble of SETAR-Trees for global time series forecasting."""

import random

import numpy as np

from aeon.forecasting.base import BaseForecaster

from ._setartree import SETARTree

__maintainer__ = ["TinaJin0228"]
__all__ = ["SETARForest"]


class SETARForest(BaseForecaster):
    """
    SETAR-Forest: An ensemble of SETAR-Trees for global time series forecasting.

    This implementation is based on the paper "SETAR-Tree: a novel and accurate
    tree algorithm for global time series forecasting" by Godahewa, R., et al. (2023).

    A SETAR-Forest consists of a collection of diverse SETAR-Trees. Diversity is
    achieved by training each tree on a bootstrapped sample of the data and by
    randomizing the hyperparameters of each tree.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of SETAR-Trees to build in the forest.
    bagging_fraction : float, default=0.8
        The fraction of the training instances (time series windows) to use for
        training each tree.
    lag : int, default=10
        The number of past lags to use as features for forecasting.
    horizon : int, default=1
        The number of time steps ahead to forecast.
    max_depth : int, default=10
        The maximum depth of each tree.
    """

    _tags = {
        "capability:exogenous": True,
    }

    def __init__(
        self,
        n_estimators: int = 10,
        bagging_fraction: float = 0.8,
        lag: int = 10,
        horizon: int = 1,
        max_depth: int = 10,
    ):
        super().__init__(horizon=horizon, axis=1)
        self.n_estimators = n_estimators
        self.bagging_fraction = bagging_fraction
        self.lag = lag
        self.max_depth = max_depth

        # This will store the individual fitted trees
        self.estimators_ = []

    def _fit(self, y, exog=None):
        """
        Fit the forest by building and training a collection of diverse SETAR-Trees.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn the forecaster.
        exog : np.ndarray, default=None
            Optional exogenous time series data. In the context of global models,
            this holds the panel of other time series.
        """
        # Combine y and exog to form the full panel of training series
        if exog is not None:
            full_panel = np.vstack([y, exog])
        else:
            full_panel = y.reshape(1, -1) if y.ndim == 1 else y

        self.estimators_ = []
        for _ in range(self.n_estimators):
            # Bagging: Create a bootstrap sample of the series
            n_series = full_panel.shape[0]
            sample_size = int(n_series * self.bagging_fraction)
            # Randomly select series indices with replacement for the sample
            sample_indices = np.random.choice(n_series, size=sample_size, replace=True)
            bootstrap_panel = full_panel[sample_indices, :]

            # The BaseForecaster expects one series in `y` and the rest in `exog`
            y_sample = bootstrap_panel[0]
            exog_sample = bootstrap_panel[1:] if bootstrap_panel.shape[0] > 1 else None

            # Hyperparameter Randomization
            # Define ranges for randomization as described in the paper,
            # can be tuned further.
            random_significance = random.uniform(0.01, 0.1)
            random_significance_divider = random.uniform(1.5, 3.0)
            random_error_threshold = random.uniform(0.01, 0.05)

            # Fit a single SETAR-tree
            tree = SETARTree(
                lag=self.lag,
                horizon=self.horizon,
                max_depth=self.max_depth,
                stopping_criteria="both",
                significance=random_significance,
                significance_divider=random_significance_divider,
                error_threshold=random_error_threshold,
            )
            tree.fit(y_sample, exog=exog_sample)
            self.estimators_.append(tree)

        return self

    def _predict(self, y=None, exog=None):
        """
        Make a prediction by averaging the forecasts from all trees in the forest.

        Parameters
        ----------
        y : np.ndarray, default=None
            The time series history to use for making a prediction.
        exog : np.ndarray, default=None
            Not used by this forest's predict method directly, as each tree
            operates on a single series history.

        Returns
        -------
        float
            The averaged forecast from all trees at the specified horizon.
        """
        self._check_is_fitted()
        if not self.estimators_:
            raise RuntimeError("The forest has not been fitted yet.")

        predictions = []
        for tree in self.estimators_:
            # Each tree makes its own prediction based on the provided history
            pred = tree.predict(y=y)
            predictions.append(pred)

        # The final forecast is the average of all individual tree forecasts
        return np.mean(predictions)
