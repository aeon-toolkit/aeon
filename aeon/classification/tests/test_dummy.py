# -*- coding: utf-8 -*-
"""Test function of DummyClassifier."""
import numpy as np

from aeon.classification import DummyClassifier
from aeon.datasets import load_unit_test


def test_dummy_classifier():
    """Test function for DummyClassifier."""
    X_train, y_train = load_unit_test(split="train")
    X_test, _ = load_unit_test(split="test")
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    assert all(i == "1" for i in pred)
    pred_proba = dummy.predict_proba(X_test)
    assert all(np.array_equal([0.5, 0.5], i) for i in pred_proba)
