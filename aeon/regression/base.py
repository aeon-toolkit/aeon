"""
Abstract base class for time series regressors.

    class name: BaseRegressor

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__all__ = [
    "BaseRegressor",
]
__author__ = ["MatthewMiddlehurst", "TonyBagnll", "mloning", "fkiraly"]
import time
from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseCollectionEstimator
from aeon.utils.sklearn import is_sklearn_transformer


class BaseRegressor(BaseCollectionEstimator, ABC):
    """Abstract base class for time series regressors.

    The base regressor specifies the methods and method signatures that all
    regressors have to implement. Attributes with a underscore suffix are set in the
    method fit.

    Parameters
    ----------
    fit_time_ : int
        Time (in milliseconds) for fit to run.
    _class_dictionary : dict
        Dictionary mapping classes_ onto integers 0...n_classes_-1.
    _n_jobs : int, default =1
        Number of threads to use in fit as determined by n_jobs.
    """

    _tags = {
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    def __init__(self):
        self._estimator_type = "regressor"

        super().__init__()

    def __rmul__(self, other):
        """Magic * method, return concatenated RegressorPipeline, transformers on left.

        Overloaded multiplication operation for regressors. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of `other` (first) with `self` (last).
        """
        from aeon.regression.compose import RegressorPipeline
        from aeon.transformations.adapt import TabularToSeriesAdaptor
        from aeon.transformations.base import BaseTransformer
        from aeon.transformations.compose import TransformerPipeline

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        if isinstance(other, BaseTransformer):
            # RegressorPipeline already has the dunder method defined
            if isinstance(self, RegressorPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return RegressorPipeline(regressor=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a RegressorPipeline
            else:
                return RegressorPipeline(regressor=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    @final
    def fit(self, X, y) -> BaseCollectionEstimator:
        """Fit time series regressor to training data.

        Parameters
        ----------
        X : np.ndarray
            train data of shape ``(n_instances, n_channels, n_timepoints)`` for any
            number of channels, equal length series, ``(n_instances, n_timepoints)``
            for univariate, equal length series.
            or list of shape ``[n_instances]`` of 2D np.array shape ``(n_channels,
            n_timepoints_i)``, where n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.
        y : np.ndarray
            1D np.array of float, of shape ``(n_instances)`` - regression targets or
            fitting indices correspond to instance indices in X.

        Returns
        -------
        BaseCollectionEstimator
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.reset()
        _start_time = int(round(time.time() * 1000))
        X = self._preprocess_collection(X)
        y = self._check_y(y, self.metadata_["n_cases"])
        self._fit(X, y)
        self.fit_time_ = int(round(time.time() * 1000)) - _start_time
        # this should happen last
        self._is_fitted = True
        return self

    @final
    def predict(self, X) -> np.ndarray:
        """Predicts target variable for time series in X.

        Parameters
        ----------
        X : np.ndarray
            train data of shape ``(n_instances, n_channels, n_timepoints)`` for any
            number of channels, equal length series, ``(n_instances, n_timepoints)``
            for univariate, equal length series.
            or list of shape ``[n_instances]`` of 2D np.array shape ``(n_channels,
            n_timepoints_i)``, where n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.

        Returns
        -------
        np.ndarray
            1D np.array of float, of shape (n_instances) - predicted regression labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()
        X = self._preprocess_collection(X)
        return self._predict(X)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : np.ndarray
            train data of shape ``(n_instances, n_channels, n_timepoints)`` for any
            number of channels, equal length series, ``(n_instances, n_timepoints)``
            for univariate, equal length series.
            or list of shape ``[n_instances]`` of 2D np.array shape ``(n_channels,
            n_timepoints_i)``, where n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.
        y : np.ndarray
            1D np.array of float, of shape ``(n_instances)`` - regression targets or
            fitting indices correspond to instance indices in X.

        Returns
        -------
        float, R-squared score of predict(X) vs y
        """
        from sklearn.metrics import r2_score

        self.check_is_fitted()
        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)
        y = y.astype("float")
        return r2_score(y, self.predict(X))

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series regressor to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_type")
            if self.get_tag("X_inner_type") = "numpy3D":
                3D np.ndarray of shape = (n_instances, n_channels, n_timepoints)
        y : 1D np.array of float, of shape (n_instances) - regression labels for
        fitting indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_"
        """
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_type")
            if self.get_tag("X_inner_type") = "numpy3D":
                3D np.ndarray of shape = (n_instances, n_channels, n_timepoints)

        Returns
        -------
        y : 1D np.array of float, of shape (n_instances) - predicted regression labels
            indices correspond to instance indices in X
        """
        ...

    def _check_y(self, y, n_cases):
        # Check y valid input for regression
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        if isinstance(y, np.ndarray) and y.ndim > 1:
            raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")
        # Check matching number of labels
        n_labels = len(y)
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                f"{n_labels}"
            )
        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)
        if isinstance(y[0], str):
            raise ValueError(
                "y contains strings, cannot fit a regressor. If suitable, convert "
                "to string."
            )
        return y
