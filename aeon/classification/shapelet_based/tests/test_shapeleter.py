"""Tests for Shapeleter classifier."""

import numpy as np

from aeon.classification.shapelet_based._shapeleter import (
    ShapeleterClassifier,
)


def test_shapeleter_classifier_fit_predict():
    """Test fit and predict functionality of ShapeleterClassifier."""
    X = np.random.RandomState(42).normal(size=(10, 1, 30))
    y = np.array([0, 1] * 5)

    clf = ShapeleterClassifier(max_shapelets=20, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert len(y_pred) == 10
    assert set(np.unique(y_pred)).issubset(set(np.unique(y)))
