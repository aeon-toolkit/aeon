"""Dummy series transformer."""

__maintainer__ = ["baraline"]
__all__ = ["DummySeriesTransformer", "DummySeriesTransformer_no_fit"]

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class DummySeriesTransformer(BaseSeriesTransformer):
    """
    DummySeriesTransformer adds a random value and a constant to the input series.

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

    def __init__(self, constant=0, axis=1, random_state=None):
        self.constant = constant
        self.random_state = random_state
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        """Simulate a fit on X by generating a random value to be added to inputs.

        Parameters
        ----------
        X : ignored argument for interface compatibility

        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        rng = np.random.RandomState(seed=self.random_state)
        self.random_value_ = rng.random()
        return self

    def _transform(self, X, y=None):
        """Transform X by adding the constant and random value set during fit.

        Parameters
        ----------
        X : np.ndarray
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray
            2D transformed version of X
        """
        return X + (self.constant + self.random_value_)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X by substracting the constant and random value.

        Parameters
        ----------
        X : np.ndarray
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray
            2D inverse transformed version of X
        """
        return X - (self.constant + self.random_value_)

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


class DummySeriesTransformer_no_fit(BaseSeriesTransformer):
    """
    DummySeriesTransformer adds a value to all elements the input series.

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

    def __init__(self, constant=0, axis=1):
        self.constant = constant
        super().__init__(axis=axis)

    def _transform(self, X, y=None):
        """Transform X by adding the constant value given in init.

        Parameters
        ----------
        X : np.ndarray
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray
            2D transformed version of X
        """
        return X + self.constant

    def _inverse_transform(self, X, y=None):
        """Inverse transform X by substracting the constant.

        Parameters
        ----------
        X : np.ndarray
            2D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray
            2D inverse transformed version of X
        """
        return X - self.constant

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
