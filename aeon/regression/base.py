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

__maintainer__ = []
__all__ = [
    "BaseRegressor",
]

import time
from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd
from deprecated.sphinx import deprecated
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target

from aeon.base import BaseCollectionEstimator
from aeon.base._base import _clone_estimator
from aeon.performance_metrics.forecasting import mean_squared_error
from aeon.utils.sklearn import is_sklearn_transformer


class BaseRegressor(BaseCollectionEstimator, ABC):
    """Abstract base class for time series regressors.

    The base regressor specifies the methods and method signatures that all
    regressors have to implement. Attributes with a underscore suffix are set in the
    method fit.

    Attributes
    ----------
    fit_time_ : int
        Time (in milliseconds) for fit to run.
    _n_jobs : int
        Number of threads to use in fit as determined by n_jobs.

    fit_time_ : int
        Time (in milliseconds) for ``fit`` to run.
    _n_jobs : int
        Number of threads to use in estimator methods such as ``fit`` and ``predict``.
        Determined by the ``n_jobs`` parameter if present.
    _estimator_type : string
        The type of estimator. Required by some ``sklearn`` tools, set to "regressor".
    """

    _tags = {
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(self):
        # reserved attributes written to in fit
        self.fit_time_ = -1
        self._n_jobs = 1

        # required for compatibility with some sklearn interfaces
        self._estimator_type = "regressor"

        super().__init__()

    # TODO: remove in v0.9.0
    @deprecated(
        version="0.8.0",
        reason="The BaseRegressor __rmul__ (*) functionality will be removed "
        "in v0.9.0.",
        category=FutureWarning,
    )
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
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If `self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseRegressor
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        start = int(round(time.time() * 1000))
        X, y = self._fit_setup(X, y)

        self._fit(X, y)

        self.fit_time_ = int(round(time.time() * 1000)) - start
        # this should happen last
        self._is_fitted = True
        return self

    @final
    def predict(self, X) -> np.ndarray:
        """Predicts target variable for time series in X.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``
            other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If `self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()
        X = self._preprocess_collection(X)
        return self._predict(X)

    @final
    def fit_predict(self, X, y) -> np.ndarray:
        """Fits the regressor and predicts class labels for X.

        fit_predict produces prediction estimates using just the train data.
        By default, this is through 10x cross validation, although some estimators may
        utilise specialist techniques such as out-of-bag estimates or leave-one-out
        cross-validation.

        Regressors which override _fit_predict will have the
        ``capability:train_estimate`` tag set to True.

        Generally, this will not be the same as fitting on the whole train data
        then making train predictions. To do this, you should call fit(X,y).predict(X)

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If `self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        X, y = self._fit_setup(X, y)

        y_pred = self._fit_predict(X, y)

        # this should happen last
        self._is_fitted = True
        return y_pred

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : np.ndarray or list
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If `self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series, so either ``n_channels == 1`` is
            true or X is 2D of shape ``(n_cases, n_timepoints)``. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability for is
            passed.
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        score : float
            MSE score of predict(X) vs y
        """
        self.check_is_fitted()
        y = self._check_y(y, len(X))
        return mean_squared_error(y, self.predict(X))

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series regressor to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets for
            fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseRegressor
            Reference to self.

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
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        ...

    def _fit_predict(self, X, y) -> np.ndarray:
        """Fits and predicts labels for sequences in X.

        Parameters
        ----------
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``
        y : np.ndarray
            1D np.array of float, of shape ``(n_cases)`` - regression targets
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            1D np.array of float, of shape (n_cases) - predicted regression labels
            indices correspond to instance indices in X
        """
        # fit the regressor
        self._fit(X, y)

        # predict using cross-validation
        random_state = getattr(self, "random_state", None)
        estimator = _clone_estimator(self, random_state)

        return cross_val_predict(
            estimator,
            X=X,
            y=y,
            cv=10,
            method="predict",
            n_jobs=self._n_jobs,
        )

    def _fit_setup(self, X, y):
        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_collection(X)
        y = self._check_y(y, self.metadata_["n_cases"])

        # return processed X and y
        return X, y

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
                f"Mismatch in number of cases. Found X = {n_cases} and y = {n_labels}"
            )

        y_type = type_of_target(y)
        if y_type != "continuous" and y_type != "binary" and y_type != "multiclass":
            raise ValueError(
                f"y type is {y_type} which is not valid for regression. "
                f"Should be continuous, binary or multiclass according to "
                f"sklearn.utils.multiclass.type_of_target"
            )

        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)

        if any([isinstance(label, str) for label in y]):
            raise ValueError(
                "y contains strings, cannot fit a regressor. If suitable, convert "
                "to floats or consider classification."
            )

        return y.astype(float)
