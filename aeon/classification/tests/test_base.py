# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality."""

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst", "achieveordie"]

import numpy as np
import pandas as pd
import pytest

from aeon.classification import DummyClassifier
from aeon.classification.base import BaseClassifier
from aeon.utils._testing.panel import _make_classification_y, _make_panel


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


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("unequal", [True, False])
def test_check_capabilities(missing, multivariate, unequal):
    """Test the checking of capabilities."""
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


def test__check_classifier_input():
    """Test for valid estimator format.

    1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    2. Test correct: X: pd.DataFrame with 1 and 3 cols vs y:np.array and np.Series
    3. Test incorrect: X with fewer cases than y
    4. Test incorrect: y as a list
    5. Test incorrect: too few cases or too short a series
    """
    cls = DummyClassifier()
    # 1. Test correct: X: np.array of 2 and 3 dimensions vs y:np.array and np.Series
    test_X1 = np.random.uniform(-1, 1, size=(5, 10))
    test_X2 = np.random.uniform(-1, 1, size=(5, 2, 10))
    test_y1 = np.random.randint(0, 1, size=5)
    test_y2 = pd.Series(np.random.randn(5))
    cls._check_classifier_input(test_X2)
    cls._check_classifier_input(test_X2, test_y1)
    cls._check_classifier_input(test_X2, test_y2)
    # 2. Test correct: X: pd.DataFrame with 1 (univariate) and 3 cols(multivariate) vs
    # y:np.array and np.Series
    test_X3 = _create_example_dataframe(5, 1, 10)
    test_X4 = _create_example_dataframe(5, 3, 10)
    cls._check_classifier_input(test_X3, test_y1)
    cls._check_classifier_input(test_X4, test_y1)
    cls._check_classifier_input(test_X3, test_y2)
    cls._check_classifier_input(test_X4, test_y2)
    # 3. Test incorrect: X with fewer cases than y
    test_X5 = np.random.uniform(-1, 1, size=(3, 4, 10))
    with pytest.raises(ValueError, match=r".*Mismatch in number of cases*."):
        cls._check_classifier_input(test_X5, test_y1)
    # 4. Test incorrect data type: y is a List
    test_y3 = [1, 2, 3, 4, 5]
    with pytest.raises(
        TypeError, match=r".*X is not of a supported input data " r"type.*"
    ):
        cls._check_classifier_input(test_X1, test_y3)
    # 5. Test incorrect: too few cases or too short a series
    with pytest.raises(ValueError, match=r".*Minimum number of cases required*."):
        cls._check_classifier_input(test_X2, test_y1, enforce_min_instances=6)


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


MTYPES = ["numpy3D", "pd-multiindex", "df-list", "numpyflat"]


@pytest.mark.parametrize("mtype", MTYPES)
def test_input_conversion_fit_predict(mtype):
    """Test that base class lets all valid input types through."""
    y = _make_classification_y()
    X = _make_panel(return_mtype=mtype)

    clf = DummyClassifier()
    clf.fit(X, y)
    clf.predict(X)

    clf = _DummyClassifier()
    clf.fit(X, y)
    clf.predict(X)


def test_predict_single_class():
    """Test return of predict predict_proba in case only single class seen in fit."""
    trainX = np.ones(shape=(10, 20))
    y = np.ones(10)
    testX = np.ones(shape=(10, 20))
    clf = DummyClassifier()
    clf.fit(trainX, y)
    y_pred = clf.predict(testX)
    y_pred_proba = clf.predict_proba(testX)
    assert y_pred.ndim == 1
    assert y_pred.shape == (10,)
    assert all(list(y_pred == 1))
    assert y_pred_proba.ndim == 2
    assert y_pred_proba.shape == (10, 1)
    assert all(list(y_pred_proba == 1))
