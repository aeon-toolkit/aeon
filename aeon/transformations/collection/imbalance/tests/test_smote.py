"""Test function for SMOTE."""

import numpy as np

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import SMOTE


def test_smote_balances_classes():
    """SMOTE oversamples the minority class up to the majority count."""
    majority, minority = 90, 10
    X = np.random.RandomState(0).rand(majority + minority, 1, 10)
    y = np.array([0] * majority + [1] * minority)

    res_X, res_y = SMOTE(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert res_X.shape == (2 * majority, 1, 10)
    assert res_y.shape == (2 * majority,)
    # every class is raised to the majority count
    assert set(counts) == {majority}


def test_smote_matches_imblearn_reference():
    """SMOTE reproduces the imbalanced-learn SMOTE output for a fixed seed.

    The expected synthetic samples were captured from
    ``imblearn.over_sampling.SMOTE(k_neighbors=1, random_state=49)`` on this input;
    this port reproduces them exactly, so the hardcoded array replaces the former
    runtime dependency on imbalanced-learn as a parity oracle. Locks the SMOTE
    interpolation ``s_i + u * (s_nn - s_i)`` and the labelling of new samples.
    """
    X, _ = make_example_3d_numpy(
        n_cases=8, n_channels=1, n_timepoints=4, random_state=0
    )
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # class 0 minority (3), class 1 majority (5)
    n_original = len(y)
    n_synthetic = 5 - 3  # majority count - minority count

    res_X, res_y = SMOTE(n_neighbors=1, random_state=49).fit_transform(X, y)

    expected = np.array(
        [
            [3.39643756, 1.79313415, 2.81692439, 2.47419297],
            [3.71802842, 1.61110021, 3.06255009, 2.22250514],
        ]
    )
    synthetic = res_X[n_original:].squeeze(axis=1)
    assert synthetic.shape == (n_synthetic, 4)
    np.testing.assert_allclose(synthetic, expected, atol=1e-6)
    # synthetic samples all belong to the oversampled minority class
    assert np.all(res_y[n_original:] == 0)


def test_smote_skips_already_balanced_class():
    """A non-majority class already at the majority count gets no synthetic samples.

    With two classes tied for the largest count, the tied non-majority class has a
    sampling target of zero and must be skipped, while the genuinely smaller class is
    still oversampled to the majority count.
    """
    counts_in = {0: 5, 1: 5, 2: 3}
    y = np.array([0] * 5 + [1] * 5 + [2] * 3)
    X = np.random.RandomState(1).rand(len(y), 1, 4)

    _, res_y = SMOTE(n_neighbors=1, random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    majority = max(counts_in.values())
    assert set(counts) == {majority}
