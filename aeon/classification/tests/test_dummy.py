"""Test function of DummyClassifier."""

import numpy as np
import pytest

from aeon.classification import DummyClassifier


@pytest.mark.parametrize(
    "strategy", ["most_frequent", "prior", "stratified", "uniform", "constant"]
)
def test_dummy_classifier_strategies(strategy):
    """Test DummyClassifier strategies."""
    X = np.ones(shape=(10, 10))
    y_train = np.random.choice([0, 1], size=10)

    dummy = DummyClassifier(strategy=strategy, constant=1)
    dummy.fit(X, y_train)

    pred = dummy.predict(X)
    assert isinstance(pred, np.ndarray)
    assert all(i in [0, 1] for i in pred)


def test_dummy_classifier_default():
    """Test DummyClassifier predicts majority class and prior distribution."""
    X = np.ones(shape=(10, 10))
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    y_expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    dummy = DummyClassifier()
    dummy.fit(X, y_train)

    pred = dummy.predict(X)
    np.testing.assert_array_equal(y_expected, pred)

    pred_proba = dummy.predict_proba(X)
    assert all(np.array_equal([0.2, 0.8], i) for i in pred_proba)
