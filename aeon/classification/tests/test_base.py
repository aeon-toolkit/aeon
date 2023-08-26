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
    get_type,
)

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst", "achieveordie"]


class _TestClassifier(BaseClassifier):
    """Classifier for testing base class fit/predict/predict_proba."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))


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


def _assert_incorrect_input(dummy, correctX, correcty, X, y, msg):
    with pytest.raises(TypeError, match=msg):
        dummy.fit(X, y)
    dummy.fit(correctX, correcty)
    with pytest.raises(TypeError, match=msg):
        dummy.predict(X)
    with pytest.raises(TypeError, match=msg):
        dummy.predict_proba(X)


def test_incorrect_input():
    """Test informative errors raised with wrong X and/or y.

    Errors are raise in aeon/utils/validation/collection.py and tested again here.
    """
    dummy = _TestClassifier()
    correctX = np.random.random(size=(5, 1, 10))
    correcty = np.array([0, 0, 1, 1, 1])
    X = ["list", "of", "string", "invalid"]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    m1 = r"ERROR passed a list containing <class 'str'>"
    m2 = r"ERROR passed input of type <class 'dict'>"
    m3 = r"y must be a np.array or a pd.Series, but found type: <class 'list'>"
    m4 = r"Mismatch in number of cases"
    m5 = r"y must be 1-dimensional"
    m6 = r"y type is continuous which is not valid for classification"
    _assert_incorrect_input(dummy, correctX, correcty, X, y, m1)
    X = {"dict": 0, "is": "not", "valid": True}
    _assert_incorrect_input(dummy, correctX, correcty, X, y, m2)
    X = np.random.random(size=(5, 1, 10))
    y = ["cannot", "pass", "list", "for", "y"]
    with pytest.raises(TypeError, match=m3):
        dummy.fit(X, y)
    # Test size mismatch
    y = np.array([0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match=m4):
        dummy.fit(X, y)
    # Multivariate y
    y = np.ndarray([0, 0, 1, 1, 1, 1])
    with pytest.raises(TypeError, match=m5):
        dummy.fit(X, y)
    # Multivariate y
    y = np.array([[0, 0], [1, 1], [1, 1]])
    with pytest.raises(TypeError, match=m5):
        dummy.fit(X, y)
    # Continuous y
    y = np.random.random(5)
    with pytest.raises(ValueError, match=m6):
        dummy.fit(X, y)


def test_checkX():
    """Test if capabilities correctly tested."""
    dummy1 = _TestClassifier()
    dummy2 = _TestHandlesAllInput()
    X = np.random.random(size=(5, 1, 10))
    assert dummy1._check_X(X) and dummy2._check_X(X)
    X[3][0][6] = np.NAN
    assert dummy2._check_X(X)
    with pytest.raises(ValueError, match=r"cannot handle missing values"):
        dummy1._check_X(X)
    X = np.random.random(size=(5, 3, 10))
    assert dummy2._check_X(X)
    with pytest.raises(ValueError, match=r"cannot handle multivariate"):
        dummy1._check_X(X)
    X[2][2][6] = np.NAN
    assert dummy2._check_X(X)
    with pytest.raises(
        ValueError, match=r"cannot handle missing values or multivariate"
    ):
        dummy1._check_X(X)
    X = [np.random.random(size=(1, 10)), np.random.random(size=(1, 20))]
    assert dummy2._check_X(X)
    with pytest.raises(ValueError, match=r"cannot handle unequal length series"):
        dummy1._check_X(X)
    X = ["Does", "Not", "Accept", "List", "of", "String"]
    with pytest.raises(TypeError, match=r"passed a list containing <class 'str'>"):
        dummy1._check_X(X)


class _MutableClassifier(BaseClassifier):
    """Classifier for testing with different internal_types."""

    def _fit(self, X, y):
        """Fit dummy."""
        return self

    def _predict(self, X):
        """Predict dummy."""
        return np.zeros(shape=(len(X),))


@pytest.mark.parametrize("internal_type", COLLECTIONS_DATA_TYPES)
@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_convertX(internal_type, data):
    """Test conversion function.

    The conversion functionality of convertCollection is tested in the utils module.
    This test runs a subset of these but also checks classifiers with multiple
    internal types.
    """
    cls = _MutableClassifier()
    # Equal length should default to numpy3D
    X = EQUAL_LENGTH_UNIVARIATE[data]
    cls.metadata_ = cls._check_X(X)
    X2 = cls._convert_X(X)
    assert get_type(X2) == cls.get_tag("X_inner_mtype")
    # Add the internal_type tag to cls, should still all revert to numpy3D
    cls.set_tags(**{"X_inner_mtype": ["numpy3D", internal_type]})
    X2 = cls._convert_X(X)
    assert get_type(X2) == "numpy3D"
    # Set cls inner type to just internal_type, should convert to internal_type
    cls.set_tags(**{"X_inner_mtype": internal_type})
    X2 = cls._convert_X(X)
    assert get_type(X2) == internal_type
    # Set to single type but in a list
    cls.set_tags(**{"X_inner_mtype": [internal_type]})
    X2 = cls._convert_X(X)
    assert get_type(X2) == internal_type
    # Set to the lowest priority data type, should convert to internal_type
    cls.set_tags(**{"X_inner_mtype": ["nested_univ", internal_type]})
    X2 = cls._convert_X(X)
    assert get_type(X2) == internal_type
    if data in UNEQUAL_LENGTH_UNIVARIATE.keys():
        if internal_type in UNEQUAL_LENGTH_UNIVARIATE.keys():
            cls.set_tags(**{"capability:unequal_length": True})
            cls.set_tags(**{"X_inner_mtype": ["nested_univ", "np-list", internal_type]})
            X = UNEQUAL_LENGTH_UNIVARIATE[data]
            X2 = cls._convert_X(X)
            assert get_type(X2) == "np-list"


def test__check_y():
    """Test private method _check_y."""
    # Correct outcomes
    cls = _TestClassifier()
    y = np.random.randint(0, 4, 100, dtype=int)
    cls._check_y(y, 100)
    assert len(cls.classes_) == cls.n_classes_ == len(cls._class_dictionary) == 4
    y = pd.Series(y)
    cls._check_y(y, 100)
    assert len(cls.classes_) == cls.n_classes_ == len(cls._class_dictionary) == 4
    # Test error raising
    # y wrong length
    with pytest.raises(ValueError, match=r"Mismatch in number of cases"):
        cls._check_y(y, 99)
    # y invalid type
    y = ["This", "is", "tested", "lots"]
    with pytest.raises(TypeError, match=r"np.array or a pd.Series"):
        cls._check_y(y, 4)
    y = np.ndarray([1, 2, 1, 2, 1, 2])
    with pytest.raises(TypeError, match=r"y must be 1-dimensional"):
        cls._check_y(y, 6)
    y = np.random.rand(10)
    with pytest.raises(ValueError, match=r"Should be binary or multiclass"):
        cls._check_y(y, 10)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_unequal_length_input(data):
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
def test_equal_length_input(data):
    """Test with unequal length failures and passes."""
    dummy = _TestClassifier()
    X = EQUAL_LENGTH_UNIVARIATE[data]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    _assert_fit_predict(dummy, X, y)
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


def test__predict_proba():
    """Test default _predict_proba."""
    cls = _TestClassifier()
    X = np.random.random(size=(5, 1, 10))
    y = np.array([1, 0, 1, 0, 1])
    with pytest.raises(KeyError):
        cls._predict_proba(X)
    cls.fit(X, y)
    p = cls._predict_proba(X)
    assert p.shape == (5, 2)
