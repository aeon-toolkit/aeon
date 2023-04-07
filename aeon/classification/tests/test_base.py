# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality."""

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst", "achieveordie"]

import pickle

import numpy as np
import pandas as pd
import pytest

from aeon.classification.base import BaseClassifier
from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.utils._testing.panel import (
    _make_classification_y,
    _make_panel,
    make_classification_problem,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


class _DummyClassifier(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self

    def _predict_proba(self, X):
        """Predict proba dummy."""
        return self


class _DummyComposite(_DummyClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    def __init__(self, foo):
        self.foo = foo


class _DummyDeepClassifierEmpty(BaseDeepClassifier):
    """Dummy Deep Classifier for testing empty base deep class save utilities."""

    def __init__(self):
        super(_DummyDeepClassifierEmpty, self).__init__()

    def build_model(self, input_shape, n_classes, **kwargs):
        return None

    def _fit(self, X, y):
        return self


class _DummyDeepClassifierFull(BaseDeepClassifier):
    """Dummy Deep Classifier to test serialization capabilities."""

    def __init__(
        self,
        optimizer,
    ):
        super(_DummyDeepClassifierFull, self).__init__()
        self.optimizer = optimizer

    def build_model(self, input_shape, n_classes, **kwargs):
        return None

    def _fit(self, X, y):
        return self


class _DummyHandlesAllInput(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self

    def _predict_proba(self, X):
        """Predict proba dummy."""
        return self


class _DummyConvertPandas(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    _tags = {
        "X_inner_mtype": "numpy3D",  # which type do _fit/_predict, support for X?
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return self

    def _predict_proba(self, X):
        """Predict proba dummy."""
        return self


multivariate_message = r"multivariate series"
missing_message = r"missing values"
unequal_message = r"unequal length series"
incorrect_X_data_structure = r"must be a np.array or a pd.Series"
incorrect_y_data_structure = r"must be 1-dimensional"


def test_base_classifier_fit():
    """Test function for the BaseClassifier class fit.

    Test fit. It should:
    1. Work with 2D, 3D and DataFrame for X and nparray for y.
    2. Calculate the number of classes and record the fit time.
    3. have self.n_jobs set or throw  an exception if the classifier can
    multithread.
    4. Set the class dictionary correctly.
    5. Set is_fitted after a call to _fit.
    6. Return self.
    """
    dummy = _DummyClassifier()
    cases = 5
    length = 10
    test_X1 = np.random.uniform(-1, 1, size=(cases, length))
    test_X2 = np.random.uniform(-1, 1, size=(cases, 2, length))
    test_X3 = _create_example_dataframe(cases=cases, dimensions=1, length=length)
    test_X4 = _create_example_dataframe(cases=cases, dimensions=3, length=length)
    test_y1 = np.random.randint(0, 2, size=(cases))
    result = dummy.fit(test_X1, test_y1)
    assert result is dummy
    with pytest.raises(ValueError, match=multivariate_message):
        result = dummy.fit(test_X2, test_y1)
    assert result is dummy
    result = dummy.fit(test_X3, test_y1)
    assert result is dummy
    with pytest.raises(ValueError, match=multivariate_message):
        result = dummy.fit(test_X4, test_y1)
    assert result is dummy
    # Raise a specific error if y is in a 2D matrix (1,cases)
    test_y2 = np.array([test_y1])
    # What if y is in a 2D matrix (cases,1)?
    test_y2 = np.array([test_y1]).transpose()
    with pytest.raises(ValueError, match=incorrect_y_data_structure):
        result = dummy.fit(test_X1, test_y2)
    # Pass a data fram
    with pytest.raises(ValueError, match=incorrect_X_data_structure):
        result = dummy.fit(test_X1, test_X3)


TF = [True, False]


@pytest.mark.parametrize("missing", TF)
@pytest.mark.parametrize("multivariate", TF)
@pytest.mark.parametrize("unequal", TF)
def test_check_capabilities(missing, multivariate, unequal):
    """Test the checking of capabilities.

    There are eight different combinations to be tested with a classifier that can
    handle it and that cannot. Obvs could loop, but I think its clearer to just
    explicitly test;
    """
    handles_none = _DummyClassifier()
    handles_none_composite = _DummyComposite(_DummyClassifier())

    # checks that errors are raised
    if missing:
        with pytest.raises(ValueError, match=missing_message):
            handles_none._check_capabilities(missing, multivariate, unequal)
    if multivariate:
        with pytest.raises(ValueError, match=multivariate_message):
            handles_none._check_capabilities(missing, multivariate, unequal)
    if unequal:
        with pytest.raises(ValueError, match=unequal_message):
            handles_none._check_capabilities(missing, multivariate, unequal)
    if not missing and not multivariate and not unequal:
        handles_none._check_capabilities(missing, multivariate, unequal)

    if missing:
        with pytest.warns(UserWarning, match=missing_message):
            handles_none_composite._check_capabilities(missing, multivariate, unequal)
    if multivariate:
        with pytest.warns(UserWarning, match=multivariate_message):
            handles_none_composite._check_capabilities(missing, multivariate, unequal)
    if unequal:
        with pytest.warns(UserWarning, match=unequal_message):
            handles_none_composite._check_capabilities(missing, multivariate, unequal)
    if not missing and not multivariate and not unequal:
        handles_none_composite._check_capabilities(missing, multivariate, unequal)

    handles_all = _DummyHandlesAllInput()
    handles_all._check_capabilities(missing, multivariate, unequal)


def test_convert_input():
    """Test the conversions from dataframe to numpy.

    1. Pass a 2D numpy X, get a 3D numpy X
    2. Pass a 3D numpy X, get a 3D numpy X
    """

    def _internal_convert(X, y=None):
        return BaseClassifier._internal_convert(None, X, y)

    cases = 5
    length = 10
    channels = 4
    test_X1 = np.ones(shape=(cases, length))
    test_X2 = np.ones(shape=(cases, channels, length))
    tester = _DummyClassifier()

    tempX = tester._convert_X(test_X1)
    assert tempX.shape == (cases, 1, length)
    tempX = tester._convert_X(test_X2)
    assert tempX.shape == (cases, channels, length)


def test__check_classifier_input():
    """Test for valid estimator format.

    1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    2. Test correct: X: pd.DataFrame with 1 and 3 cols vs y:np.array and np.Series
    3. Test incorrect: X with fewer cases than y
    4. Test incorrect: y as a list
    5. Test incorrect: too few cases or too short a series
    """

    def _check_classifier_input(X, y=None, enforce_min_instances=1):
        return BaseClassifier._check_classifier_input(None, X, y, enforce_min_instances)

    # 1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    test_X2 = np.random.uniform(-1, 1, size=(5, 2, 10))
    test_y1 = np.random.randint(0, 1, size=5)
    test_y2 = pd.Series(np.random.randn(5))
    _check_classifier_input(test_X2)
    _check_classifier_input(test_X2, test_y1)
    _check_classifier_input(test_X2, test_y2)
    # 2. Test correct: X: pd.DataFrame with 1 (univariate) and 3 cols(multivariate) vs
    # y:np.array and np.Series
    test_X3 = _create_example_dataframe(5, 1, 10)
    test_X4 = _create_example_dataframe(5, 3, 10)
    _check_classifier_input(test_X3, test_y1)
    _check_classifier_input(test_X4, test_y1)
    _check_classifier_input(test_X3, test_y2)
    _check_classifier_input(test_X4, test_y2)
    # 3. Test incorrect: X with fewer cases than y
    test_X5 = np.random.uniform(-1, 1, size=(3, 4, 10))
    with pytest.raises(ValueError, match=r".*Mismatch in number of cases*."):
        _check_classifier_input(test_X5, test_y1)
    # 4. Test incorrect data type: y is a List
    test_y3 = [1, 2, 3, 4, 5]
    with pytest.raises(
        TypeError, match=r".*X is not of a supported input data " r"type.*"
    ):
        _check_classifier_input(test_X1, test_y3)
    # 5. Test incorrect: too few cases or too short a series
    with pytest.raises(ValueError, match=r".*Minimum number of cases required*."):
        _check_classifier_input(test_X2, test_y1, enforce_min_instances=6)


def _create_example_dataframe(cases=5, dimensions=1, length=10):
    """Create a simple data frame set of time series (X) for testing."""
    test_X = pd.DataFrame(dtype=np.float32)
    for i in range(0, dimensions):
        instance_list = []
        for _ in range(0, cases):
            instance_list.append(pd.Series(np.random.randn(length)))
        test_X["dimension_" + str(i)] = instance_list
    return test_X


def _create_unequal_length_nested_dataframe(cases=5, dimensions=1, length=10):
    testy = pd.DataFrame(dtype=np.float32)
    for i in range(0, dimensions):
        instance_list = []
        for _ in range(0, cases - 1):
            instance_list.append(pd.Series(np.random.randn(length)))
        instance_list.append(pd.Series(np.random.randn(length - 1)))
        testy["dimension_" + str(i + 1)] = instance_list

    return testy


MTYPES = ["numpy3D", "pd-multiindex", "df-list", "numpyflat", "nested_univ"]


@pytest.mark.parametrize("mtype", MTYPES)
def test_input_conversion_fit_predict(mtype):
    """Test that base class lets all Panel mtypes through."""
    y = _make_classification_y()
    X = _make_panel(return_mtype=mtype)

    clf = Catch22Classifier()
    clf.fit(X, y)
    clf.predict(X)

    clf = _DummyConvertPandas()
    clf.fit(X, y)
    clf.predict(X)


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_predict_single_class(method):
    """Test return of predict/_proba in case only single class seen in fit."""
    X, y = make_classification_problem()
    y[:] = 42
    n_instances = 10
    X_test = X[:n_instances]

    clf = KNeighborsTimeSeriesClassifier()

    clf.fit(X, y)
    y_pred = getattr(clf, method)(X_test)

    if method == "predict":
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.ndim == 1
        assert y_pred.shape == (n_instances,)
        assert all(list(y_pred == 42))
    if method == "predict_proba":
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.ndim == 2
        assert y_pred.shape == (n_instances, 1)
        assert all(list(y_pred == 1))


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_deep_estimator_empty():
    """Check if serialization works for empty dummy."""
    empty_dummy = _DummyDeepClassifierEmpty()
    serialized_empty = pickle.dumps(empty_dummy)
    deserialized_empty = pickle.loads(serialized_empty)
    assert empty_dummy.__dict__ == deserialized_empty.__dict__


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("optimizer", [None, "adam", "object-adamax"])
def test_deep_estimator_full(optimizer):
    """Check if serialization works for full dummy."""
    from tensorflow.keras.optimizers import Adamax, Optimizer, serialize

    if optimizer == "object-adamax":
        optimizer = Adamax()

    full_dummy = _DummyDeepClassifierFull(optimizer)
    serialized_full = pickle.dumps(full_dummy)
    deserialized_full = pickle.loads(serialized_full)

    if isinstance(optimizer, Optimizer):
        # assert same configuration of optimizer
        assert serialize(full_dummy.__dict__["optimizer"]) == serialize(
            deserialized_full.__dict__["optimizer"]
        )
        assert serialize(full_dummy.optimizer) == serialize(deserialized_full.optimizer)

        # assert weights of optimizers are same
        assert (
            full_dummy.optimizer.variables() == deserialized_full.optimizer.variables()
        )

        # remove optimizers from both to do full dict check,
        # since two different objects
        del full_dummy.__dict__["optimizer"]
        del deserialized_full.__dict__["optimizer"]

    # check if components are same
    assert full_dummy.__dict__ == deserialized_full.__dict__
