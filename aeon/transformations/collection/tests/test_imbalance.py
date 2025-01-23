"""Tests for the rebalancer transformers."""

import numpy as np
import pytest

from aeon.transformations.collection.imbalance import SMOTE, ADASYN


def test_smote():
    """Test the SMOTE class.

    This function creates a 3D numpy array, applies
    SMOTE using the SMOTE class, and asserts that the
    transformed data has a balanced number of samples.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)

    transformer = SMOTE()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert len(res_X) == 2 * majority_num
    assert len(res_y) == 2 * majority_num
    assert res_count[0] == majority_num
    assert res_count[1] == majority_num


def test_adasyn():
    """Test the ADASYN class.

    This function creates a 3D numpy array, applies
    ADASYN using the ADASYN class, and asserts that the
    transformed data has a balanced number of samples.
    ADASYN is a variant of SMOTE that generates synthetic samples,
    but it focuses on generating samples near the decision boundary.
    Therefore, sometimes, it may generate more or less samples than SMOTE,
    which is why we only check if the number of samples is nearly balanced.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)

    transformer = ADASYN()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert np.abs(len(res_X) - 2 * majority_num) < minority_num
    assert np.abs(len(res_y) - 2 * majority_num) < minority_num
    assert res_count[0] == majority_num
    assert np.abs(res_count[0] - res_count[1]) < minority_num
