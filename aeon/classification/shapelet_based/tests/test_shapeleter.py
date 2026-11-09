"""Tests for Shapeleter classifier."""

import numpy as np

from aeon.classification.shapelet_based._shapeleter import ShapeleterClassifier
from aeon.datasets import load_unit_test


def test_shapeleter_classifier_4views_and_voters():
    """Test that all 4 representation transformers and 4 ensemble voters are fitted."""
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


def test_shapeleter_classifier_predict_proba_vote_shares():
    """Test that predict_proba returns vote shares that strictly agree with predict."""
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
    probas = clf.predict_proba(X)

    assert probas.shape == (8, 2)
    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    # In 4-voter hard voting, probabilities are multiples of 0.25
    valid_shares = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for val in probas.ravel():
        assert np.any(np.isclose(val, valid_shares, atol=1e-5))

    # Predicted class must match argmax of probas (ties resolved to lower class index)
    predicted_indices = np.array([list(clf.classes_).index(p) for p in preds])
    np.testing.assert_array_equal(np.argmax(probas, axis=1), predicted_indices)


def test_shapeleter_classifier_unit_test():
    """Test ShapeleterClassifier on UnitTest dataset to verify stability."""
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    clf = ShapeleterClassifier(
        max_shapelets=50,
        per_shapelets=10,
        max_iter=5,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    assert preds.shape == (len(X_test),)
    assert probas.shape == (len(X_test), len(clf.classes_))
    predicted_indices = np.array([list(clf.classes_).index(p) for p in preds])
    np.testing.assert_array_equal(np.argmax(probas, axis=1), predicted_indices)
