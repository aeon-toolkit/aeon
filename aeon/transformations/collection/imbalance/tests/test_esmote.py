"""Test function for ESMOTE."""

import numpy as np

from aeon.transformations.collection.imbalance import ESMOTE


def test_esmote_balances_classes():
    """ESMOTE oversamples the minority class up to the majority count."""
    majority, minority = 20, 6
    X = np.random.RandomState(0).rand(majority + minority, 1, 10)
    y = np.array([0] * majority + [1] * minority)

    res_X, res_y = ESMOTE(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert res_y.shape == (2 * majority,)
    assert set(counts) == {majority}


def test_esmote_deterministic():
    """A fixed random_state gives identical synthetic series across fits.

    ESMOTE draws neighbours, step sizes and alignment tie-breaks from its random
    state; seeding it must make the whole elastic generation reproducible.
    """
    X = np.random.RandomState(1).rand(26, 1, 10)
    y = np.array([0] * 20 + [1] * 6)

    res1 = ESMOTE(random_state=7).fit_transform(X, y)[0]
    res2 = ESMOTE(random_state=7).fit_transform(X, y)[0]

    np.testing.assert_array_equal(res1, res2)


def test_esmote_skips_already_balanced_class():
    """A non-majority class already at the majority count gets no synthetic samples."""
    y = np.array([0] * 8 + [1] * 8 + [2] * 6)
    X = np.random.RandomState(2).rand(len(y), 1, 10)

    _, res_y = ESMOTE(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert set(counts) == {8}  # smaller class raised to the tied-majority count
