"""Unit tests for the prevalidated ridge classifier."""

import numpy as np
import pytest

from aeon.classification.sklearn import PrevalidatedRidgeClassifier


def test_prevalidated_ridge_classifier_lifecycle_binary():
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

    clf = PrevalidatedRidgeClassifier(lambdas=lambdas)
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
    assert clf.coef_.shape == (2, X.shape[1])
    np.testing.assert_array_equal(clf.coef_[:, clf.mask_], 0.0)
    assert clf.intercept_.shape == (2,)
    np.testing.assert_array_equal(clf.classes_, np.array(["a", "b"]))


def test_prevalidated_ridge_classifier_lifecycle_multiclass():
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

    clf = PrevalidatedRidgeClassifier(lambdas=np.logspace(-2, 2, 5).astype(np.float32))
    clf.fit(X, y)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 3)
    assert set(preds).issubset(set(y))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_array_equal(preds, clf.classes_[np.argmax(proba, axis=1)])
    assert clf.coef_.shape == (3, X.shape[1])
    assert clf.intercept_.shape == (3,)
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))


def test_prevalidated_ridge_classifier_n_lt_p_with_low_variance_columns():
    """Test the high-dimensional path and removal of low-variance columns."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 10)).astype(np.float32)
    X[:, 0] = 1.0
    X[:, 1] = np.linspace(0.0, 1e-7, X.shape[0], dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    clf = PrevalidatedRidgeClassifier(lambdas=np.array([0.1, 1.0], dtype=np.float32))
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    np.testing.assert_array_equal(
        clf.predict(X), clf.classes_[np.argmax(proba, axis=1)]
    )
    np.testing.assert_array_equal(clf.mask_[:2], [True, True])
    assert not np.any(clf.mask_[2:])
    assert clf.n_cases_ == X.shape[0]
    assert clf.n_atts_ == X.shape[1]
    assert clf.coef_.shape == (2, X.shape[1])
    np.testing.assert_array_equal(clf.coef_[:, clf.mask_], 0.0)
    assert clf.intercept_.shape == (2,)


def test_prevalidated_ridge_classifier_public_coefficients_reproduce_proba():
    """Test that sklearn-style fitted attributes reproduce probabilities."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 5)).astype(np.float32)
    X[:, 2] = 3.0
    y = np.array([0, 1, 2] * 4)

    clf = PrevalidatedRidgeClassifier(lambdas=np.array([0.1, 1.0])).fit(X, y)

    logits = X @ clf.coef_.T + clf.intercept_
    log_eps = np.log(np.finfo(np.float32).eps)
    exp_logits = np.exp(logits.clip(log_eps, -log_eps))
    expected = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(clf.predict_proba(X), expected, atol=1e-6)


def test_prevalidated_ridge_classifier_against_reference_implementation():
    """Test probabilities against hard-coded output from the reference PreVal."""
    X = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 2, 0, 1, 2])
    X_test = np.array(
        [[0.25, 0.75, 2.0], [1.5, 0.25, 2.0], [2.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    lambdas = np.logspace(-2, 2, 5).astype(np.float32)

    # Generated with https://github.com/angus924/preval as follows:
    # from preval import PreVal
    # reference = PreVal(lambdas=lambdas)
    # reference.fit(X, y)
    # print(repr(reference.predict_proba(X_test)))
    reference_proba = np.array(
        [
            [1.0800920e-08, 1.0392226e-04, 9.9989605e-01],
            [9.9787033e-01, 2.1251689e-03, 4.5259776e-06],
            [9.9999517e-01, 4.8196589e-06, 2.3229221e-11],
        ],
        dtype=np.float32,
    )

    clf = PrevalidatedRidgeClassifier(lambdas=lambdas).fit(X, y)

    np.testing.assert_allclose(clf.predict_proba(X_test), reference_proba, rtol=1e-5)


@pytest.mark.parametrize("lambdas", [[], [0.0, 1.0], [-1.0, 1.0], [1.0, np.inf]])
def test_prevalidated_ridge_classifier_invalid_lambdas(lambdas):
    """Test invalid lambda grids are rejected early."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    y = np.array(["a", "a", "b", "b"])

    with pytest.raises(ValueError, match="lambdas must contain"):
        PrevalidatedRidgeClassifier(lambdas=lambdas).fit(X, y)
