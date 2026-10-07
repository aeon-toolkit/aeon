"""Test ADASYN oversampler."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import ADASYN


def test_adasyn_balances_classes_approximately():
    """ADASYN oversamples the minority class to roughly the majority count.

    Unlike SMOTE, ADASYN allocates samples by local difficulty and rounds the
    per-sample counts, so the minority class is only approximately balanced.
    """
    majority, minority = 90, 10
    X = np.random.RandomState(0).rand(majority + minority, 1, 10)
    y = np.array([0] * majority + [1] * minority)

    res_X, res_y = ADASYN(random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert counts[0] == majority  # majority class untouched
    assert abs(counts[0] - counts[1]) < minority  # minority near-balanced


def test_adasyn_matches_imblearn_reference():
    """ADASYN reproduces the imbalanced-learn ADASYN output for a fixed seed.

    The expected synthetic samples were captured from
    ``imblearn.over_sampling.ADASYN(n_neighbors=1, random_state=49)`` on this input;
    this port reproduces them exactly, so the hardcoded array replaces the former
    runtime dependency on imbalanced-learn as a parity oracle. Locks the density
    ratio allocation and the synthetic-sample construction.
    """
    X, _ = make_example_3d_numpy(
        n_cases=8, n_channels=1, n_timepoints=4, random_state=0
    )
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # class 0 minority (3), class 1 majority (5)
    n_original = len(y)

    res_X, res_y = ADASYN(n_neighbors=1, random_state=49).fit_transform(X, y)

    expected = np.array(
        [
            [0.60192891, 1.77745026, 1.36949884, 1.83535333],
            [2.45821348, 2.32420838, 2.10032461, 3.20847871],
            [2.13662262, 2.50624231, 1.85469891, 3.46016654],
        ]
    )
    synthetic = res_X[n_original:].squeeze(axis=1)
    np.testing.assert_allclose(synthetic, expected, atol=1e-6)
    assert np.all(res_y[n_original:] == 0)


def test_adasyn_skips_already_balanced_class():
    """A non-majority class already at the majority count gets no synthetic samples."""
    y = np.array([0] * 5 + [1] * 5 + [2] * 3)
    X = np.random.RandomState(1).rand(len(y), 1, 4)

    _, res_y = ADASYN(n_neighbors=1, random_state=0).fit_transform(X, y)
    _, counts = np.unique(res_y, return_counts=True)

    assert counts[0] == 5 and counts[1] == 5  # tied majorities untouched
    assert counts[2] > 3  # smaller class was oversampled


def test_adasyn_raises_when_no_majority_neighbours():
    """ADASYN raises if no minority neighbour is from the majority class.

    When the minority class is tightly clustered and well separated from the
    majority, every neighbour of a minority point is itself a minority point, so the
    density ratio is all zero and ADASYN cannot allocate samples.
    """
    X_min = 0.01 * np.random.RandomState(2).rand(4, 1, 4)
    X_maj = 100 + np.random.RandomState(3).rand(8, 1, 4)
    X = np.vstack([X_min, X_maj])
    y = np.array([0] * 4 + [1] * 8)

    with pytest.raises(RuntimeError, match="majority"):
        ADASYN(n_neighbors=3, random_state=0).fit_transform(X, y)


def test_adasyn_raises_when_rounding_yields_no_samples():
    """ADASYN raises if the density-ratio rounding allocates zero samples.

    With a sampling target of a single sample spread across many minority points,
    every per-point allocation rounds down to zero, so no synthetic samples can be
    produced and ADASYN reports this rather than returning an unchanged set.
    """
    rs = np.random.RandomState(0)
    X = np.vstack([rs.rand(8, 1, 6), rs.rand(9, 1, 6) + 0.3])  # target = 9 - 8 = 1
    y = np.array([0] * 8 + [1] * 9)

    with pytest.raises(ValueError, match="No samples will be generated"):
        ADASYN(n_neighbors=5, random_state=0).fit_transform(X, y)
