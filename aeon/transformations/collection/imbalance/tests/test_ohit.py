"""Test function for OHIT."""

import numpy as np

from aeon.transformations.collection.imbalance import OHIT


def test_ohit():
    """Test the OHIT class.

    This function creates a 3D numpy array, applies
    OHIT using the OHIT class, and asserts that the
    transformed data has a balanced number of samples.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)

    transformer = OHIT()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert len(res_X) == 2 * majority_num
    assert len(res_y) == 2 * majority_num
    assert res_count[0] == majority_num
    assert res_count[1] == majority_num
