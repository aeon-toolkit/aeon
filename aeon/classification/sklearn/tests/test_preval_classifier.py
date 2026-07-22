"""Smoke tests for the PreVal classifier."""

import numpy as np
import pytest

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
    result = clf.fit(X, y)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 2)
    assert set(preds).issubset(set(y))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_array_equal(preds, clf.classes_[np.argmax(proba, axis=1)])
    assert result is clf
    assert clf.is_fitted
    assert clf.lambda_ in lambdas
    assert clf.scale_.shape == ()
    assert clf.mask_.shape == (X.shape[1],)
    np.testing.assert_array_equal(clf.mask_, [False, False, True])
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (2,)
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
    y = np.array([0, 1, 2, 0, 1, 2])

    clf = PreValClassifier(lambdas=np.logspace(-2, 2, 5).astype(np.float32))
    clf.fit(X, y)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 3)
    assert set(preds).issubset(set(y))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_array_equal(preds, clf.classes_[np.argmax(proba, axis=1)])
    assert clf.coef_.shape == (X.shape[1] + 1, 3)
    assert clf.intercept_.shape == (3,)
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))


def test_preval_classifier_n_lt_p_with_low_variance_columns():
    """Test the high-dimensional path and removal of low-variance columns."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 10)).astype(np.float32)
    X[:, 0] = 1.0
    X[:, 1] = np.linspace(0.0, 1e-7, X.shape[0], dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    clf = PreValClassifier(lambdas=np.array([0.1, 1.0], dtype=np.float32))
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    np.testing.assert_array_equal(
        clf.predict(X), clf.classes_[np.argmax(proba, axis=1)]
    )
    np.testing.assert_array_equal(clf.mask_[:2], [True, True])
    assert not np.any(clf.mask_[2:])
    assert clf.n_cases_ == X.shape[0]
    assert clf.n_atts_ == X.shape[1]
    assert clf.coef_.shape == (X.shape[1] - 2 + 1, 2)
    assert clf.intercept_.shape == (2,)


@pytest.mark.parametrize("lambdas", [[], [0.0, 1.0], [-1.0, 1.0], [1.0, np.inf]])
def test_preval_classifier_invalid_lambdas(lambdas):
    """Test invalid lambda grids are rejected early."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    y = np.array(["a", "a", "b", "b"])

    with pytest.raises(ValueError, match="lambdas must contain"):
        PreValClassifier(lambdas=lambdas).fit(X, y)
