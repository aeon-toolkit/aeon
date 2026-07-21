"""Smoke tests for the PreVal classifier."""

import numpy as np

from aeon.classification.sklearn import PreValClassifier


def test_preval_classifier_lifecycle_binary():
    """Test the full estimator lifecycle on a small binary problem."""
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.1, 1.0],
            [0.9, 0.8, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array(["a", "a", "b", "b", "a", "b"])
    lambdas = np.logspace(-2, 2, 5).astype(np.float32)

    clf = PreValClassifier(lambdas=lambdas)
    clf.fit(X, y)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 2)
    assert set(preds).issubset(set(y))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert clf.lambda_ in lambdas
    np.testing.assert_array_equal(clf.classes_, np.array(["a", "b"]))


def test_preval_classifier_lifecycle_multiclass():
    """Test the full estimator lifecycle on a small multiclass problem."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array(["c0", "c1", "c2", "c0", "c1", "c2"])

    clf = PreValClassifier(lambdas=np.logspace(-2, 2, 5).astype(np.float32))
    clf.fit(X, y)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 3)
    assert set(preds).issubset(set(y))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_array_equal(clf.classes_, np.array(["c0", "c1", "c2"]))


def test_preval_classifier_invalid_lambdas():
    """Test invalid lambda grids are rejected early."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    y = np.array(["a", "a", "b", "b"])

    for lambdas in ([], [0.0, 1.0], [-1.0, 1.0], [1.0, np.inf]):
        try:
            PreValClassifier(lambdas=lambdas).fit(X, y)
        except ValueError:
            continue

        raise AssertionError(f"Expected ValueError for lambdas={lambdas!r}")
