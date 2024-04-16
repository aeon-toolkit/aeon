"""Mock series transformers."""

import numpy as np

from aeon.transformations.series import BaseSeriesTransformer


class MockUnivariateSeriesTransformer(BaseSeriesTransformer):
    """
    MockSeriesTransformer adds a random value and a constant to the input series.

    Parameters
    ----------
    constant : int, default=0
        The constant to be added to each element of the input time series.
    random_state : int or None, default=None
        Seed for random number generation.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, constant: int = 0, random_state=None) -> None:
        self.constant = constant
        self.random_state = random_state
        super().__init__(axis=1)

    def _fit(self, X: np.ndarray, y=None):
        """Simulate a fit on X by generating a random value to be added to inputs.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        self.n_features_ = X.shape[0]
        rng = np.random.RandomState(seed=self.random_state)
        self.random_values_ = rng.random(self.n_features_)
        return self

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform X by adding the constant and random value set during fit.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray, shape = (n_channels, n_timepoints)
            2D transformed version of X
        """
        X_new = np.zeros_like(X)
        for i in range(self.n_features_):
            X_new[i] = X[i] + (self.constant + self.random_values_[i])
        return X_new

    def _inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Inverse transform X by substracting the constant and random value.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray, shape = (n_channels, n_timepoints)
            2D inverse transformed version of X
        """
        X_new = np.zeros_like(X)
        for i in range(self.n_features_):
            X_new[i] = X[i] - (self.constant + self.random_values_[i])
        return X_new

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {}


class MockMultivariateSeriesTransformer(BaseSeriesTransformer):
    """
    MockSeriesTransformer adds a random value and a constant to the input series.

    Parameters
    ----------
    constant : int, default=0
        The constant to be added to each element of the input time series.
    random_state : int or None, default=None
        Seed for random number generation.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:inverse_transform": True,
    }

    def __init__(self, constant: int = 0, random_state=None) -> None:
        self.constant = constant
        self.random_state = random_state
        super().__init__(axis=1)

    def _fit(self, X: np.ndarray, y=None):
        """Simulate a fit on X by generating a random value to be added to inputs.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        self.n_features_ = X.shape[0]
        rng = np.random.RandomState(seed=self.random_state)
        self.random_values_ = rng.random(self.n_features_)
        return self

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform X by adding the constant and random value set during fit.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray, shape = (n_channels, n_timepoints)
            2D transformed version of X
        """
        X_new = np.zeros_like(X)
        for i in range(self.n_features_):
            X_new[i] = X[i] + (self.constant + self.random_values_[i])
        return X_new

    def _inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Inverse transform X by substracting the constant and random value.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray, shape = (n_channels, n_timepoints)
            2D inverse transformed version of X
        """
        X_new = np.zeros_like(X)
        for i in range(self.n_features_):
            X_new[i] = X[i] - (self.constant + self.random_values_[i])
        return X_new

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {}


class MockSeriesTransformerNoFit(BaseSeriesTransformer):
    """
    MockSeriesTransformerNoFit adds a value to all elements the input series.

    Parameters
    ----------
    constant : int, default=0
        The constant to be added to each element of the input time series.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:inverse_transform": True,
        "fit_is_empty": True,
    }

    def __init__(self, constant: int = 0) -> None:
        self.constant = constant
        super().__init__(axis=1)

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform X by adding the constant value given in init.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray, shape = (n_channels, n_timepoints)
            2D transformed version of X
        """
        X_new = np.zeros_like(X)
        X_new = X + self.constant
        return X_new

    def _inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Inverse transform X by substracting the constant.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray, shape = (n_channels, n_timepoints)
            2D inverse transformed version of X
        """
        X_new = np.zeros_like(X)
        X_new = X - self.constant
        return X_new

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {}
