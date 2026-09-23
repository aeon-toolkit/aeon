"""Test function for OHIT."""

import numpy as np

from aeon.transformations.collection.imbalance import OHIT


def test_ohit_balances_classes():
    """OHIT oversamples the minority class up to the majority count."""
    majority, minority = 90, 10
    X = np.random.RandomState(0).rand(majority + minority, 1, 10)
    y = np.array([0] * majority + [1] * minority)

    res_X, res_y = OHIT(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert res_X.shape == (2 * majority, 1, 10)
    assert set(counts) == {majority}


def test_ohit_random_state_reproducible():
    """Same random_state gives identical output."""
    X = np.random.RandomState(0).rand(100, 1, 10)
    y = np.array([0] * 90 + [1] * 10)

    res1 = OHIT(random_state=49).fit_transform(X, y)[0]
    res2 = OHIT(random_state=49).fit_transform(X, y)[0]

    assert np.array_equal(res1, res2)


def test_ohit_does_not_mutate_params():
    """fit_transform leaves k and kapa as set in the constructor."""
    X = np.random.RandomState(0).rand(100, 1, 10)
    y = np.array([0] * 90 + [1] * 10)

    transformer = OHIT(random_state=49)
    transformer.fit_transform(X, y)

    assert transformer.get_params()["k"] is None
    assert transformer.get_params()["kapa"] is None


def test_ohit_single_sample_minority_class():
    """A minority class with a single sample is oversampled by replication.

    DRSNN clustering needs more than one point, so OHIT falls back to tiling the
    lone minority series up to the required count.
    """
    y = np.array([0] * 1 + [1] * 10 + [2] * 6)
    X = np.random.RandomState(5).rand(len(y), 1, 8)
    lone = X[0]

    res_X, res_y = OHIT(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert set(counts) == {10}
    # every class-0 sample is a copy of the single original minority series
    class0 = res_X[res_y == 0]
    assert np.all(class0 == lone)


def test_ohit_no_cluster_fallback():
    """When DRSNN finds no core points, all minority samples form one cluster.

    Forcing the density-ratio threshold above any achievable value leaves no core
    points, so OHIT must fall back to treating the whole minority class as a single
    cluster and still return a balanced set.
    """
    majority, minority = 20, 6
    X = np.random.RandomState(4).rand(majority + minority, 1, 8)
    y = np.array([0] * majority + [1] * minority)

    _, res_y = OHIT(drT=1e9, random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert set(counts) == {majority}


def test_ohit_skips_already_balanced_class():
    """A non-majority class already at the majority count gets no synthetic samples."""
    y = np.array([0] * 20 + [1] * 20 + [2] * 10)
    X = np.random.RandomState(6).rand(len(y), 1, 8)

    _, res_y = OHIT(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert set(counts) == {20}
