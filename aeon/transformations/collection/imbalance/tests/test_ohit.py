"""Test function for OHIT."""

import numpy as np

from aeon.transformations.collection.imbalance import OHIT


def _make_imbalanced_data(n_samples=100, majority_num=90, random_state=0):
    rng = np.random.RandomState(random_state)
    X = rng.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * (n_samples - majority_num))
    return X, y


def test_ohit():
    """Test the OHIT class.

    This function creates a 3D numpy array, applies
    OHIT using the OHIT class, and asserts that the
    transformed data has a balanced number of samples.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class

    X, y = _make_imbalanced_data(n_samples, majority_num)

    transformer = OHIT()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert len(res_X) == 2 * majority_num
    assert len(res_y) == 2 * majority_num
    assert res_count[0] == majority_num
    assert res_count[1] == majority_num


def test_ohit_random_state_is_reproducible():
    """Test OHIT returns the same samples for the same random state."""
    X, y = _make_imbalanced_data()

    res_X1, res_y1 = OHIT(random_state=49).fit_transform(X, y)
    res_X2, res_y2 = OHIT(random_state=49).fit_transform(X, y)

    np.testing.assert_array_equal(res_X1, res_X2)
    np.testing.assert_array_equal(res_y1, res_y2)


def test_ohit_default_k_and_kapa_do_not_mutate_params():
    """Test fitted defaults do not overwrite constructor parameters."""
    X_large, y_large = _make_imbalanced_data()
    X_small, y_small = _make_imbalanced_data(n_samples=20, majority_num=16)
    transformer = OHIT(random_state=49)

    transformer.fit_transform(X_large, y_large)

    assert transformer.get_params()["k"] is None
    assert transformer.get_params()["kapa"] is None

    transformer.fit_transform(X_small, y_small)

    assert transformer.get_params()["k"] is None
    assert transformer.get_params()["kapa"] is None
