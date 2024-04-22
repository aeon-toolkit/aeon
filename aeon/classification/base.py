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
__maintainer__ = []

import time
from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target

from aeon.base import BaseCollectionEstimator
from aeon.base._base import _clone_estimator
from aeon.utils.validation._dependencies import _check_estimator_deps
from aeon.utils.validation.collection import get_n_cases


class BaseClassifier(BaseCollectionEstimator, ABC):
    """
    Abstract base class for time series classifiers.

    The base classifier specifies the methods and method signatures that all
    classifiers have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels, either integers or strings.
    n_classes_ : int
        Number of classes (length of ``classes_``).
    fit_time_ : int
        Time (in milliseconds) for ``fit`` to run.
    _class_dictionary : dict
        Mapping of classes_ onto integers ``0 ... n_classes_-1``.
    _n_jobs : int
        Number of threads to use in estimator methods such as ``fit`` and ``predict``.
        Determined by the ``n_jobs`` parameter if present.
    _estimator_type : string
        The type of estimator. Required by some ``sklearn`` tools, set to "classifier".
    """

    _tags = {
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(self):
        # reserved attributes written to in fit
        self.classes_ = []  # classes seen in y, unique labels
        self.n_classes_ = -1  # number of unique classes in y
        self._class_dictionary = {}
        self.fit_time_ = -1
        self._n_jobs = 1

        # required for compatibility with some sklearn interfaces e.g.
        # CalibratedClassifierCV
        self._estimator_type = "classifier"

        super().__init__()
        _check_estimator_deps(self)

    @final
    def fit(self, X, y) -> BaseCollectionEstimator:
        """Fit time series classifier to training data.

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
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseClassifier
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        start = int(round(time.time() * 1000))
        X, y, single_class = self._fit_setup(X, y)

        if not single_class:
            self._fit(X, y)

        self.fit_time_ = int(round(time.time() * 1000)) - start
        # this should happen last
        self._is_fitted = True
        return self

    @final
    def predict(self, X) -> np.ndarray:
        """Predicts class labels for time series in X.

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
            1D np.array of float, of shape (n_cases) - predicted class labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            n_cases = get_n_cases(X)
            return np.repeat(list(self._class_dictionary.keys()), n_cases)

        X = self._preprocess_collection(X)
        return self._predict(X)

    @final
    def predict_proba(self, X) -> np.ndarray:
        """Predicts class label probabilities for time series in X.

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

        Returns
        -------
        probabilities : np.ndarray
            2D array of shape ``(n_cases, n_classes)`` - predicted class probabilities
            First dimension indices correspond to instance indices in X,
            second dimension indices correspond to class labels, (i, j)-th entry is
            estimated probability that i-th instance is of class j
        """
        self.check_is_fitted()

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            n_cases = get_n_cases(X)
            return np.repeat([[1]], n_cases, axis=0)

        X = self._preprocess_collection(X)
        return self._predict_proba(X)

    @final
    def fit_predict(self, X, y) -> np.ndarray:
        """Fits the classifier and predicts class labels for X.

        fit_predict produces prediction estimates using just the train data.
        By default, this is through 10x cross validation, although some estimators may
        utilise specialist techniques such as out-of-bag estimates or leave-one-out
        cross-validation.

        Classifiers which override _fit_predict will have the
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
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            shape ``[n_cases]`` - predicted class labels indices correspond to
            instance indices in
        """
        X, y, single_class = self._fit_setup(X, y)

        if single_class:
            n_cases = get_n_cases(X)
            y_pred = np.repeat(list(self._class_dictionary.keys()), n_cases)
        else:
            y_pred = self._fit_predict(X, y)

        # this should happen last
        self._is_fitted = True
        return y_pred

    @final
    def fit_predict_proba(self, X, y) -> np.ndarray:
        """Fits the classifier and predicts class label probabilities for X.

        fit_predict_proba produces probability estimates using just the train data.
        By default, this is through 10x cross validation, although some estimators may
        utilise specialist techniques such as out-of-bag estimates or leave-one-out
        cross-validation.

        Classifiers which override _fit_predict_proba will have the
        ``capability:train_estimate`` tag set to True.

        Generally, this will not be the same as fitting on the whole train data
        then making train predictions. To do this, you should call
        fit(X,y).predict_proba(X)

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
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        probabilities : np.ndarray
            2D array of shape ``(n_cases, n_classes)`` - predicted class probabilities
            First dimension indices correspond to instance indices in X,
            second dimension indices correspond to class labels, (i, j)-th entry is
            estimated probability that i-th instance is of class j
        """
        X, y, single_class = self._fit_setup(X, y)

        if single_class:
            n_cases = get_n_cases(X)
            y_proba = np.repeat([[1]], n_cases, axis=0)
        else:
            y_proba = self._fit_predict_proba(X, y)

        # this should happen last
        self._is_fitted = True
        return y_proba

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
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        score : float
             Accuracy score of predict(X) vs y.
        """
        self.check_is_fitted()
        self._check_y(y, len(X), update_classes=False)
        return accuracy_score(y, self.predict(X), normalize=True)

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series classifier to training data.

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
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        self : BaseClassifier
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
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``

        Returns
        -------
        predictions : np.ndarray
            shape ``[n_cases]`` - predicted class labels indices correspond to
            instance indices in
        """
        ...

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

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
        probabilities : np.ndarray
            2D array of shape ``(n_cases, n_classes)`` - predicted class probabilities
            First dimension indices correspond to instance indices in X,
            second dimension indices correspond to class labels, (i, j)-th entry is
            estimated probability that i-th instance is of class j
        """
        preds = self._predict(X)
        n_pred = len(preds)
        dists = np.zeros((n_pred, self.n_classes_))
        for i in range(n_pred):
            dists[i, self._class_dictionary[preds[i]]] = 1

        return dists

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
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        predictions : np.ndarray
            shape ``[n_cases]`` - predicted class labels indices correspond to
            instance indices in
        """
        return self._fit_predict_default(X, y, "predict")

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        """Fits and predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : Train data
            guaranteed to be of a type in self.get_tag("X_inner_type")
            if ``self.get_tag("X_inner_type")`` equals "numpy3D":
                3D np.ndarray of shape ``(n_cases, n_channels, n_timepoints)``
            if ``self.get_tag("X_inner_type")`` equals "np-list":
                list of 2D np.ndarray of shape ``(n_cases)``
        y : np.ndarray
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.

        Returns
        -------
        probabilities : np.ndarray
            2D array of shape ``(n_cases, n_classes)`` - predicted class probabilities
            First dimension indices correspond to instance indices in X,
            second dimension indices correspond to class labels, (i, j)-th entry is
            estimated probability that i-th instance is of class j
        """
        return self._fit_predict_default(X, y, "predict_proba")

    def _fit_setup(self, X, y):
        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_collection(X)
        y = self._check_y(y, self.metadata_["n_cases"])

        # return processed X and y, and whether there is only one class
        return X, y, len(self.classes_) == 1

    def _check_y(self, y, n_cases, update_classes=True):
        # Check y valid input for classification
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        if isinstance(y, np.ndarray) and y.ndim > 1:
            raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")

        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Found X = {n_cases} and y = {n_labels}"
            )

        y_type = type_of_target(y)
        if y_type != "binary" and y_type != "multiclass":
            raise ValueError(
                f"y type is {y_type} which is not valid for classification. "
                f"Should be binary or multiclass according to "
                f"sklearn.utils.multiclass.type_of_target"
            )

        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)

        # remember class labels
        if update_classes:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            self._class_dictionary = {}
            for index, class_val in enumerate(self.classes_):
                self._class_dictionary[class_val] = index

        return y

    def _fit_predict_default(self, X, y, method):
        # fit the classifier
        self._fit(X, y)

        # predict using cross-validation
        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class
            if cv_size < 2:
                raise ValueError(
                    f"All classes must have at least 2 values to run the "
                    f"_fit_{method} cross-validation."
                )

        random_state = getattr(self, "random_state", None)
        estimator = _clone_estimator(self, random_state)

        return cross_val_predict(
            estimator,
            X=X,
            y=y,
            cv=cv_size,
            method=method,
            n_jobs=self._n_jobs,
        )
