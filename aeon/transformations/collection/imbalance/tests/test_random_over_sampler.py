"""Tests for RandomOverSampler."""

import numpy as np
from sklearn.utils import check_random_state

from aeon.transformations.collection.imbalance import RandomOverSampler


def test_random_over_sampler_balances_classes():
    """RandomOverSampler should bring every class to the majority count."""
    rng = check_random_state(0)
    X = rng.randn(10, 1, 12)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    X_res, y_res = RandomOverSampler(random_state=0).fit_transform(X, y)
    _, counts = np.unique(y_res, return_counts=True)
    assert X_res.shape == (14, 1, 12)
    assert y_res.shape == (14,)
    assert counts.tolist() == [7, 7]


def test_random_over_sampler_keeps_originals():
    """Original samples should still be present after oversampling."""
    rng = check_random_state(1)
    X = rng.randn(8, 1, 5)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    X_res, y_res = RandomOverSampler(random_state=1).fit_transform(X, y)
    # first n originals kept in order for majority and minority blocks
    assert np.allclose(X_res[:8], X)
    assert np.array_equal(y_res[:8], y)


def test_random_over_sampler_multivariate():
    """RandomOverSampler should accept multi-channel series and copy whole cases."""
    rng = check_random_state(2)
    X = rng.randn(9, 3, 16)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    X_res, y_res = RandomOverSampler(random_state=2).fit_transform(X, y)
    _, counts = np.unique(y_res, return_counts=True)
    assert X_res.shape == (12, 3, 16)
    assert counts.tolist() == [6, 6]
    # originals preserved at the front; all channels kept
    assert np.allclose(X_res[:9], X)
    assert np.array_equal(y_res[:9], y)


def test_random_over_sampler_multiclass_balances_to_majority():
    """Every minority class is raised to the majority count in a multi-class panel."""
    rng = check_random_state(0)
    X = rng.randn(20, 1, 15)
    y = np.array([0] * 12 + [1] * 5 + [2] * 3)

    _, y_res = RandomOverSampler(random_state=0).fit_transform(X, y)
    _, counts = np.unique(y_res, return_counts=True)

    assert counts.tolist() == [12, 12, 12]
