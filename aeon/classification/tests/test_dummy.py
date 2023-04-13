# -*- coding: utf-8 -*-
"""Test function of DummyClassifier."""
import numpy as np

from aeon.classification import DummyClassifier


def test_dummy_classifier():
    """Test DummyClassifier predicts majority class and prior distribution."""
    X_train = np.ones(shape=(10, 10))
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    y_expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X_test = np.ones(shape=(10, 10))
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    np.testing.assert_array_equal(y_expected, pred)
    pred_proba = dummy.predict_proba(X_test)

    assert all(np.array_equal([0.2, 0.8], i) for i in pred_proba)
