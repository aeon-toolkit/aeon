"""Tests for RandomOverSampler."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.transformations.collection.imbalance import RandomOverSampler
from aeon.utils.validation._dependencies import _check_soft_dependencies


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


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "imbalanced-learn",
        package_import_alias={"imbalanced-learn": "imblearn"},
        severity="none",
    ),
    reason="skip test if required soft dependency imbalanced-learn not available",
)
def test_random_over_sampler_matches_imblearn():
    """Match imblearn RandomOverSampler with sampling_strategy='all'."""
    from imblearn.over_sampling import RandomOverSampler as ImbROS

    rng = check_random_state(0)
    X2 = rng.randn(20, 15)
    y = np.array([0] * 12 + [1] * 5 + [2] * 3)
    X_imb, y_imb = ImbROS(sampling_strategy="all", random_state=0).fit_resample(X2, y)
    X_aeon, y_aeon = RandomOverSampler(random_state=0).fit_transform(
        X2[:, np.newaxis, :], y
    )
    X_aeon = X_aeon.squeeze(1)

    def sort_xy(X, y):
        y = np.asarray(y)
        idx = np.lexsort((X[:, 0], y.astype(str)))
        return X[idx], y[idx]

    Xi, yi = sort_xy(X_imb, y_imb)
    Xa, ya = sort_xy(X_aeon, y_aeon)
    assert Xi.shape == Xa.shape
    assert np.array_equal(yi.astype(str), ya.astype(str))
    assert np.allclose(Xi, Xa)
