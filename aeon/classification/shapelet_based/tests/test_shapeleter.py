"""Tests for Shapeleter classifier."""

import numpy as np

from aeon.classification.shapelet_based._shapeleter import ShapeleterClassifier


def test_shapeleter_classifier_parameter_routing():
    """Test that ShapeleterClassifier parameters route correctly."""
    clf = ShapeleterClassifier(
        max_shapelets=25,
        per_shapelets=5,
        ka=6.0,
        ko=7.0,
        max_iter=10,
        random_state=42,
    )
    assert clf.max_shapelets == 25
    assert clf.per_shapelets == 5
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
        per_shapelets=5,
        max_iter=2,
        random_state=42,
    )
    clf.fit(X, y)
    preds = clf.predict(X)

    assert len(preds) == len(y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "transformer_raw_")
    assert hasattr(clf, "transformer_t0_")
    assert hasattr(clf, "transformer_t1_")
    assert hasattr(clf, "transformer_t2_")
    assert hasattr(clf, "clf_v0_")
    assert hasattr(clf, "clf_v1_")
    assert hasattr(clf, "clf_v2_")
    assert hasattr(clf, "clf_v3_")
    assert np.all(np.isin(preds, clf.classes_))


def test_shapeleter_classifier_predict_proba():
    """Test ShapeleterClassifier predict_proba returns valid probabilities."""
    X = np.random.RandomState(42).normal(size=(8, 1, 30))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    clf = ShapeleterClassifier(
        max_shapelets=10,
        per_shapelets=5,
        max_iter=2,
        random_state=42,
    )
    clf.fit(X, y)
    probas = clf.predict_proba(X)

    assert probas.shape == (8, 2)
    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)
