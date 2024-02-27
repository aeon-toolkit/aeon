"""Unit tests for classifier base class functionality."""

import numpy as np
import numpy.random
import pandas as pd
import pytest

from aeon.testing.mock_estimators import (
    MockClassifier,
    MockClassifierFullTags,
    MockClassifierPredictProba,
)
from aeon.testing.utils.data_gen._collection import (
    EQUAL_LENGTH_MULTIVARIATE,
    EQUAL_LENGTH_UNIVARIATE,
    UNEQUAL_LENGTH_UNIVARIATE,
)
from aeon.utils import COLLECTIONS_DATA_TYPES

__maintainer__ = []


multivariate_message = r"multivariate series"
missing_message = r"missing values"
unequal_message = r"unequal length series"
incorrect_X_data_structure = r"must be a np.ndarray or a pd.Series"
incorrect_y_data_structure = r"must be 1-dimensional"


def test_incorrect_input():
    """Test informative errors raised with wrong X and/or y.

    Errors are raised in aeon/utils/validation/collection.py and tested again here.
    """
    dummy = MockClassifier()
    # dummy data to pass to fit when testing predict/predict_proba
    dummy_X = np.random.random(size=(5, 1, 10))
    dummy_y = np.array([0, 0, 1, 1, 1])

    # correct y
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Test list of str X
    m1 = r"ERROR passed a list containing <class 'str'>"
    X = ["list", "of", "string", "invalid"]
    _assert_incorrect_X_input(dummy, dummy_X, dummy_y, X, y, m1)

    # Test 2d list of str X
    m1 = r"lists should either 2D numpy arrays or pd.DataFrames"
    X = [["list", "of", "string", "invalid"] for _ in range(5)]
    _assert_incorrect_X_input(dummy, dummy_X, dummy_y, X, y, m1)

    # Test dict X
    m2 = r"ERROR passed input of type <class 'dict'>"
    X = {"dict": 0, "is": "not", "valid": True}
    _assert_incorrect_X_input(dummy, dummy_X, dummy_y, X, y, m2)

    # correct X
    X = np.random.random(size=(5, 1, 10))

    # Test list y
    m3 = r"y must be a np.array or a pd.Series, but found type: <class 'list'>"
    y = ["cannot", "pass", "list", "for", "y"]
    with pytest.raises(TypeError, match=m3):
        dummy.fit(X, y)

    # Test size mismatch
    m4 = r"Mismatch in number of cases"
    y = np.array([0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match=m4):
        dummy.fit(X, y)

    m5 = r"y must be 1-dimensional"
    # Multivariate y 1
    y = np.ndarray([0, 0, 1, 1, 1, 1])
    with pytest.raises(TypeError, match=m5):
        dummy.fit(X, y)
    # Multivariate y 2
    y = np.array([[0, 0], [1, 1], [1, 1]])
    with pytest.raises(TypeError, match=m5):
        dummy.fit(X, y)

    # Continuous y
    m6 = r"y type is continuous which is not valid for classification"
    y = np.random.random(5)
    with pytest.raises(ValueError, match=m6):
        dummy.fit(X, y)


def _assert_incorrect_X_input(dummy, correctX, correcty, X, y, msg):
    with pytest.raises(TypeError, match=msg):
        dummy.fit(X, y)

    dummy.fit(correctX, correcty)

    with pytest.raises(TypeError, match=msg):
        dummy.predict(X)
    with pytest.raises(TypeError, match=msg):
        dummy.predict_proba(X)


def test_check_y():
    """Test private method _check_y."""
    cls = MockClassifier()

    # Correct outcomes
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


@pytest.mark.parametrize("data", UNEQUAL_LENGTH_UNIVARIATE.keys())
def test_unequal_length_input(data):
    """Test with unequal length failures and passes."""
    X = UNEQUAL_LENGTH_UNIVARIATE[data]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Unable to handle unequal length series
    dummy = MockClassifier()
    with pytest.raises(ValueError, match=r"cannot handle unequal length series"):
        dummy.fit(X, y)

    # Able to handle unequal length series
    dummy = MockClassifierFullTags()
    _assert_fit_and_predict(dummy, X, y)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_univariate_equal_length_input(data):
    """Test with unequal length failures and passes."""
    X = EQUAL_LENGTH_UNIVARIATE[data]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Default capabilities
    dummy = MockClassifier()
    _assert_fit_and_predict(dummy, X, y)

    # All capabiltiies
    dummy = MockClassifierFullTags()
    _assert_fit_and_predict(dummy, X, y)


@pytest.mark.parametrize("data", EQUAL_LENGTH_MULTIVARIATE.keys())
def test_multivariate_equal_length_input(data):
    """Test with unequal length failures and passes."""
    X = EQUAL_LENGTH_MULTIVARIATE[data]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Unable to handle multivariate series
    dummy = MockClassifier()
    with pytest.raises(ValueError, match=r"cannot handle multivariate series"):
        dummy.fit(X, y)

    # Able to handle multivariate series
    dummy = MockClassifierFullTags()
    _assert_fit_and_predict(dummy, X, y)


def _assert_fit_and_predict(dummy, X, y):
    result = dummy.fit(X, y)

    # Fit returns self
    assert result is dummy

    preds = dummy.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 10

    preds = dummy.predict_proba(X)
    assert preds.shape == (10, 2)


def test_classifier_score():
    """Test the base class score() function."""
    dummy = MockClassifier()

    X = np.random.random(size=(6, 10))
    y = np.array([0, 0, 0, 1, 1, 1])
    dummy.fit(X, y)
    assert dummy.score(X, y) == 0.5

    y2 = pd.Series([0, 0, 0, 1, 1, 1])
    dummy.fit(X, y2)
    assert dummy.score(X, y) == 0.5
    assert dummy.score(X, y2) == 0.5


def test_predict_single_class():
    """Test return of predict/predict_proba in case only single class seen in fit."""
    X = np.ones(shape=(10, 20))
    y = np.ones(10)

    clf = MockClassifierPredictProba()
    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.ndim == 1
    assert y_pred.shape == (10,)
    assert all(list(y_pred == 1))

    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.ndim == 2
    assert y_pred_proba.shape == (10, 1)
    assert all(list(y_pred_proba == 1))


def test_predict_proba_default():
    """Test default _predict_proba."""
    X = np.random.random(size=(5, 1, 10))
    y = np.array([1, 0, 1, 0, 1])
    cls = MockClassifier()

    # fails if not fitted
    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        cls._predict_proba(X)

    cls.fit(X, y)

    p = cls._predict_proba(X)
    assert p.shape == (5, 2)


def test_fit_predict():
    """Test fit_predict and fit_predict_proba."""
    X = np.random.random(size=(5, 1, 10))
    y = np.array([1, 0, 1, 0, 1])
    cls = MockClassifier()

    p = cls.fit_predict(X, y)
    assert p.shape == (5,)

    p = cls.fit_predict_proba(X, y)
    assert p.shape == (5, 2)

    p = cls.predict(X)
    assert p.shape == (5,)


def test_fit_predict_single_class():
    """Test return of fit_predict/fit_predict_proba in case only single class."""
    X = np.ones(shape=(10, 20))
    y = np.ones(10)
    clf = MockClassifierPredictProba()

    y_pred = clf.fit_predict(X, y)
    assert y_pred.ndim == 1
    assert y_pred.shape == (10,)
    assert all(list(y_pred == 1))

    y_pred_proba = clf.fit_predict_proba(X, y)
    assert y_pred_proba.ndim == 2
    assert y_pred_proba.shape == (10, 1)
    assert all(list(y_pred_proba == 1))

    y_pred = clf.predict(X)
    assert y_pred.ndim == 1
    assert y_pred.shape == (10,)
    assert all(list(y_pred == 1))


def test_fit_predict_default():
    """Test fit_predict and fit_predict_proba."""
    cls = MockClassifier()

    # test default fit_predict cv size
    X = np.random.random(size=(20, 1, 10))
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    p = cls._fit_predict_default(X, y, "predict")
    assert p.shape == (20,)

    # test default fit_predict cv size
    X = np.random.random(size=(2, 1, 10))
    y = np.array([1, 0])

    with pytest.raises(ValueError, match=r"All classes must have at least 2 values"):
        cls._fit_predict_default(X, y, "predict")
