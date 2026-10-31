"""Tests for Shapeleter classifier."""

import numpy as np

from aeon.classification.shapelet_based._shapeleter import ShapeleterClassifier


def test_shapeleter_classifier_parameter_routing():
    """Test that ShapeleterClassifier parameters route correctly."""
    clf = ShapeleterClassifier(
        max_shapelets=25,
        ka=6.0,
        ko=7.0,
        max_iter=10,
        random_state=42,
    )
    assert clf.max_shapelets == 25
    assert clf.ka == 6.0
    assert clf.ko == 7.0
    assert clf.max_iter == 10
    assert clf.random_state == 42


def test_shapeleter_classifier_fit_predict():
    """Test ShapeleterClassifier fit and predict on synthetic data."""
    X = np.random.RandomState(42).normal(size=(8, 1, 30))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    clf = ShapeleterClassifier(
        max_shapelets=10,
        max_iter=5,
        random_state=42,
    )
    clf.fit(X, y)
    preds = clf.predict(X)

    assert len(preds) == len(y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "transformer_")
    assert hasattr(clf, "clf_")
    assert np.all(np.isin(preds, clf.classes_))
