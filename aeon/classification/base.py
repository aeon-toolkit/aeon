# -*- coding: utf-8 -*-
"""
Abstract base class for time series classifiers.

    class name: BaseClassifier

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)
                    - predict_proba(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__all__ = [
    "BaseClassifier",
]
__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

import time
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator
from aeon.datatypes import check_is_scitype, convert_to
from aeon.utils.sklearn import is_sklearn_transformer
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation._dependencies import _check_estimator_deps


class BaseClassifier(BaseEstimator, ABC):
    """Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Attributes
    ----------
    classes_            : ndarray of class labels, possibly strings
    n_classes_          : integer, number of classes (length of ``classes_``)
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _X_metadata         : metadata/properties of X seen in fit
    _class_dictionary   : dictionary mapping classes_ onto integers
        0...``n_classes_``-1.
    _threads_to_use     : number of threads to use in ``fit`` as determined by
    _estimator_type     : string required by sklearn, set to "classifier"
    """

    _tags = {
        "X_inner_mtype": "numpy3D",  # Currently, it should be "numpy3D". Unequal
        # length series through "nested_univ" is no longer supported, will be
        # reintroduced with lists of numpy.
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    def __init__(self):
        # reserved attributes written to in fit
        self.classes_ = []  # classes seen in y, unique labels
        self.n_classes_ = 0  # number of unique classes in y
        self.fit_time_ = 0  # time elapsed in last fit call
        self._X_metadata = []  # metadata/properties of X seen in fit
        self._class_dictionary = {}
        self._threads_to_use = 1

        # required for compatibility with some sklearn interfaces e.g.       #
        # CalibratedClassifierCV
        self._estimator_type = "classifier"

        super(BaseClassifier, self).__init__()
        _check_estimator_deps(self)

    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Overloaded multiplication operation for classifiers. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of `other` (first) with `self` (last).
        """
        from aeon.classification.compose import ClassifierPipeline
        from aeon.transformations.base import BaseTransformer
        from aeon.transformations.compose import TransformerPipeline
        from aeon.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        #  todo: this can probably be simplified further with "zero length" pipelines
        if isinstance(other, BaseTransformer):
            # ClassifierPipeline already has the dunder method defined
            if isinstance(self, ClassifierPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return ClassifierPipeline(classifier=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a ClassifierPipeline
            else:
                return ClassifierPipeline(classifier=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape [n_instances, n_channels, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        # reset estimator at the start of fit
        self.reset()

        start = int(round(time.time() * 1000))
        # convenience conversions to allow user flexibility:
        # if X is 2D array, convert to 3D, if y is Series, convert to numpy
        X, y = self._internal_convert(X, y)
        X_metadata = self._check_classifier_input(X, y)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        self._X_metadata = X_metadata

        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)

        # remember class labels
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}
        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index

        # escape early and do not fit if only one class label has been seen
        #   in this case, we later predict the single class label seen
        if len(self.classes_) == 1:
            self.fit_time_ = int(round(time.time() * 1000)) - start
            self._is_fitted = True
            return self

        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)
        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        # pass coerced and checked data to inner _fit
        self._fit(X, y)
        self.fit_time_ = int(round(time.time() * 1000)) - start

        # this should happen last
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predicts labels for time series in X.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape [n_instances, n_channels, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of channels, equal or unequal length series)

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()

        # input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict")

        # call internal _predict_proba
        return self._predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape [n_instances, n_channels, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of channels, equal or unequal length series)

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            First dimension indices correspond to instance indices in X,
            second dimension indices correspond to class labels, (i, j)-th entry is
            estimated probability that i-th instance is of class j
        """
        self.check_is_fitted()

        # input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict_proba")

        # call internal _predict_proba
        return self._predict_proba(X)

    def _single_class_y_pred(self, X, method="predict"):
        """Handle the prediction case where only single class label was seen in fit."""
        _, _, X_meta = check_is_scitype(X, scitype="Panel", return_metadata=True)
        n_instances = X_meta["n_instances"]
        if method == "predict":
            return np.repeat(list(self._class_dictionary.keys()), n_instances)
        else:  # method == "predict_proba"
            return np.repeat([[1]], n_instances, axis=0)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape [n_instances, n_channels, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
        y : 1D np.ndarray of shape [n_instances] - class labels (ground truth)
            indices correspond to instance indices in X

        Returns
        -------
        float, accuracy score of predict(X) vs y
        """
        from sklearn.metrics import accuracy_score

        self.check_is_fitted()

        return accuracy_score(y, self.predict(X), normalize=True)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
        """
        return super().get_test_params(parameter_set=parameter_set)

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_channels, series_length]
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_channels, series_length]

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        """
        ...

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_channels, series_length]

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        preds = self._predict(X)
        n_pred = len(preds)
        dists = np.zeros((n_pred, self.n_classes_))
        for i in range(n_pred):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists

    def _check_convert_X_for_predict(self, X):
        """Input checks, capability checks, repeated in all predict/score methods.

        Parameters
        ----------
        X : any object (to check/convert)
            should be of a supported Panel mtype or 2D numpy.ndarray

        Returns
        -------
        X: an object of a supported Panel mtype, numpy3D if X was a 2D numpy.ndarray

        Raises
        ------
        ValueError if X is of invalid input data type, or there is not enough data
        ValueError if the capabilities in self._tags do not handle the data.
        """
        X = self._internal_convert(X)
        X_metadata = self._check_classifier_input(X)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        # Check this classifier can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the classifier tags
        X = self._convert_X(X)

        return X

    def _check_capabilities(self, missing, multivariate, unequal):
        """Check whether this classifier can handle the data characteristics.

        Parameters
        ----------
        missing : boolean, does the data passed to fit contain missing values?
        multivariate : boolean, does the data passed to fit contain missing values?
        unequal : boolea, do the time series passed to fit have variable lengths?

        Raises
        ------
        ValueError if the capabilities in self._tags do not handle the data.
        """
        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")

        self_name = type(self).__name__

        # identify problems, mismatch of capability and inputs
        problems = []
        if missing and not allow_missing:
            problems += ["missing values"]
        if multivariate and not allow_multivariate:
            problems += ["multivariate series"]
        if unequal and not allow_unequal:
            problems += ["unequal length series"]

        # construct error message
        problems_and = " and ".join(problems)
        problems_or = " or ".join(problems)
        msg = (
            f"Data seen by {self_name} instance has {problems_and}, "
            f"but this {self_name} instance cannot handle {problems_or}. "
            f"Calls with {problems_or} may result in error or unreliable results."
        )

        # raise exception or warning with message
        # if self is composite, raise a warning, since passing could be fine
        #   see discussion in PR 2366 why
        if len(problems) > 0:
            if self.is_composite():
                warn(msg)
            else:
                raise ValueError(msg)

    def _convert_X(self, X):
        """Convert equal length series from DataFrame to numpy array or vice versa.

        Parameters
        ----------
        self : this classifier
        X : np.ndarray. Input time series.

        Returns
        -------
        X : input X converted to type in "X_inner_mtype" (3D np.ndarray)
            Checked and possibly converted input data
        """
        inner_type = self.get_tag("X_inner_mtype")
        X = convert_to(
            X,
            to_type=inner_type,
            as_scitype="Panel",
        )
        return X

    def _check_classifier_input(self, X, y=None, enforce_min_instances=1):
        """Check whether input X and y are valid formats with minimum data.

        Raises a ValueError if the input is not valid.

        Parameters
        ----------
        X : check whether X is a valid input type
        y : check whether y is a pd.Series or np.array
        enforce_min_instances : int, optional (default=1)
            check there are a minimum number of instances.

        Returns
        -------
        metadata : dict with metadata for X

        Raises
        ------
        ValueError
            If y or X is invalid input data type, or there is not enough data
        """
        # Check X is valid input type and recover the data characteristics
        X_valid, _, X_metadata = check_is_scitype(
            X, scitype="Panel", return_metadata=True
        )
        if not X_valid:
            raise TypeError(
                f"X is not of a supported input data type."
                f"X must be in a supported data type, found {type(X)}."
            )
        n_cases = X_metadata["n_instances"]
        if n_cases < enforce_min_instances:
            raise ValueError(
                f"Minimum number of cases required is {enforce_min_instances} but X "
                f"has : {n_cases}"
            )

        # Check y if passed
        if y is not None:
            # Check y valid input
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError(
                    f"y must be a np.array or a pd.Series, but found type: {type(y)}"
                )
            # Check matching number of labels
            n_labels = y.shape[0]
            if n_cases != n_labels:
                raise ValueError(
                    f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                    f"{n_labels}"
                )
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError(
                        f"np.ndarray y must be 1-dimensional, "
                        f"but found {y.ndim} dimensions"
                    )
            # warn if only a single class label is seen
            # this should not raise exception since this can occur by train subsampling
            if len(np.unique(y)) == 1:
                warn(
                    "only single class label seen in y passed to "
                    f"fit of classifier {type(self).__name__}"
                )

        return X_metadata

    def _internal_convert(self, X, y=None):
        """Convert X and y to supported types.

        Convert X to a 3D numpy array if it is a 2D and convert y into an 1D numpy
        array if passed as a pd.Series.

        Parameters
        ----------
        X : an object of any supported type
        y : np.ndarray or pd.Series

        Returns
        -------
        X: a numpy3D if X was a 2D numpy.ndarray, otherwise X is unchanged
        y: np.ndarray
        """
        if isinstance(X, np.ndarray):
            # Force 2D numpy to be 3D numpy for interface consistency.
            if X.ndim == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
        if y is not None and isinstance(y, pd.Series):
            # y should be a numpy array, although we allow Series for user convenience
            y = pd.Series.to_numpy(y)
        if y is None:
            return X
        return X, y
