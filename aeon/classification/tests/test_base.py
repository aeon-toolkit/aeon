# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality."""

import numpy as np
import numpy.random
import pandas as pd
import pytest

from aeon.classification import DummyClassifier
from aeon.classification.base import BaseClassifier, _get_metadata
from aeon.utils.validation.collection import COLLECTIONS_DATA_TYPES
from aeon.utils.validation.tests.test_collection import (
    EQUAL_LENGTH_UNIVARIATE,
    UNEQUAL_LENGTH_UNIVARIATE,
)

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst", "achieveordie"]

"""
 Need to test:
    1. base class fit, predict and predict_proba works with valid input,
    raises exception with invalid
    2. checkX and convertX, valid and invald
    3. _get_metadata
    4. _check_y
    5. score

"""


class _TestClassifier(BaseClassifier):
    """Cassifier for testing base class fit/predict/predict_proba."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))

    def _predict_proba(self, X):
        """Predict proba dummy."""
        return np.zeros(shape=(len(X), 2))


class _TestHandlesAllInput(BaseClassifier):
    """Dummy classifier for testing base class fit/predict/predict_proba."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_mtype": ["np-list", "numpy3D"],
    }

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))

    def _predict_proba(self, X):
        """Predict proba dummy."""
        return np.zeros(shape=(len(X), 2))


multivariate_message = r"multivariate series"
missing_message = r"missing values"
unequal_message = r"unequal length series"
incorrect_X_data_structure = r"must be a np.ndarray or a pd.Series"
incorrect_y_data_structure = r"must be 1-dimensional"


def _assert_fit_predict(dummy, X, y):
    result = dummy.fit(X, y)
    # Fit returns self
    assert result is dummy
    preds = dummy.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 10
    preds = dummy.predict_proba(X)
    assert preds.shape == (10, 2)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_equal_length(data):
    """Test basic functionality with valid input for the BaseClassifier."""
    dummy = _TestClassifier()
    X = EQUAL_LENGTH_UNIVARIATE[data]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    _assert_fit_predict(dummy, X, y)


def test_incorrect_input():
    """Test informative errors raised with wrong X and/or y.

    Errors are raise in aeon/utils/validation/collection.py and tested again here.
    """
    dummy = _TestClassifier()
    X = ["list", "of", "string", "invalid"]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    with pytest.raises(
        TypeError, match=r"ERROR passed a list containing <class 'str'>"
    ):
        dummy.fit(X, y)
    X = {"dict": 0, "is": "not", "valid": True}
    with pytest.raises(TypeError, match=r"ERROR passed input of type <class 'dict'>"):
        dummy.fit(X, y)
    X = np.random.random(size=(5, 1, 10))
    y = ["cannot", "pass", "list", "for", "y"]
    with pytest.raises(TypeError, match=r"found type: <class 'list'>"):
        dummy.fit(X, y)
    # Test size mismatch
    y = np.array([0, 0, 1, 1, 1, 1])
    # Multivariate y
    y = np.ndarray([0, 0, 1, 1, 1, 1])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
        dummy.fit(X, y)
    # Multivariate y
    y = np.array([[0, 0], [1, 1], [1, 1]])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
        dummy.fit(X, y)
    # Continuous y
    y = np.random.random(5)
    with pytest.raises(
        ValueError, match=r"y type is continuous which is not valid for classification"
    ):
        dummy.fit(X, y)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_unequal_length(data):
    """Test with unequal length failures and passes."""
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    if data in UNEQUAL_LENGTH_UNIVARIATE.keys():
        dummy = _TestClassifier()
        X = UNEQUAL_LENGTH_UNIVARIATE[data]
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        with pytest.raises(ValueError, match=r"cannot handle unequal length series"):
            dummy.fit(X, y)
        dummy = _TestHandlesAllInput()
        _assert_fit_predict(dummy, X, y)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test__get_metadata(data):
    """Test get meta data."""
    X = EQUAL_LENGTH_UNIVARIATE[data]
    meta = _get_metadata(X)
    assert not meta["multivariate"]
    assert not meta["missing_values"]
    assert not meta["unequal_length"]
    assert meta["n_cases"] == 10


# @pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
# def test_convertX(data):
#    """Directly test the conversions."""


def test_classifier_score():
    """Test the base class score() function."""
    X = np.random.random(size=(6, 10))
    y = np.array([0, 0, 0, 1, 1, 1])
    dummy = DummyClassifier()
    dummy.fit(X, y)
    assert dummy.score(X, y) == 0.5
    y2 = pd.Series([0, 0, 0, 1, 1, 1])
    dummy.fit(X, y2)
    assert dummy.score(X, y) == 0.5
    assert dummy.score(X, y2) == 0.5


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
