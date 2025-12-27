"""Tests for DFT-SFA mindist distance functions.

This module tests mindist_dft_sfa_distance and mindist_dft_sfa_pairwise_distance,
validating tighter lower bound properties.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.mindist import (
    mindist_dft_sfa_distance,
    mindist_dft_sfa_pairwise_distance,
    mindist_sfa_distance,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.dictionary_based import SFAWhole


def test_dft_sfa_mindist_tighter_lower_bound():
    """Test that DFT-SFA mindist >= SFA mindist (tighter bound)."""
    x, _ = make_example_3d_numpy(n_cases=1, n_channels=1, n_timepoints=10)
    y = x + 10

    transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    x_sfa, x_dft = transform.fit_transform(x)
    y_sfa, y_dft = transform.transform(y)

    for i in range(len(x_sfa)):
        # DFT-SFA mindist
        dist_dft_sfa = mindist_dft_sfa_distance(
            y_dft[i], x_sfa[i], transform.breakpoints
        )

        # SFA mindist
        dist_sfa = mindist_sfa_distance(x_sfa[i], y_sfa[i], transform.breakpoints)

        # DFT-SFA should be >= SFA mindist (tighter lower bound)
        assert dist_dft_sfa >= dist_sfa - 1e-10


def test_dft_sfa_mindist_lower_bound():
    """Test that DFT-SFA mindist is a lower bound for Euclidean."""
    x, _ = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=20)
    y, _ = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=20, random_state=42
    )

    transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    x_sfa, x_dft = transform.fit_transform(x)
    y_sfa, y_dft = transform.transform(y)

    for i in range(len(x_sfa)):
        dist_dft_sfa = mindist_dft_sfa_distance(
            x_dft[i], y_sfa[i], transform.breakpoints
        )

        euclidean_dist = np.linalg.norm(x[i] - y[i])

        assert dist_dft_sfa <= euclidean_dist + 1e-10


def test_dft_sfa_mindist_identity():
    """Test that identical representations have zero distance."""
    x, _ = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=15)

    transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    x_sfa, x_dft = transform.fit_transform(x)

    for i in range(len(x_sfa)):
        # DFT to itself with same SFA
        dist = mindist_dft_sfa_distance(x_dft[i], x_sfa[i], transform.breakpoints)

        # Should be 0 or very close
        assert dist < 1e-8


def test_dft_sfa_mindist_non_negativity():
    """Test that DFT-SFA mindist is always non-negative."""
    x, _ = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=10)
    y, _ = make_example_3d_numpy(
        n_cases=3, n_channels=1, n_timepoints=10, random_state=123
    )

    transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    x_sfa, x_dft = transform.fit_transform(x)
    y_sfa, y_dft = transform.transform(y)

    for i in range(len(x_sfa)):
        dist = mindist_dft_sfa_distance(x_dft[i], y_sfa[i], transform.breakpoints)
        assert dist >= 0


def test_dft_sfa_mindist_breakpoint_boundaries():
    """Test DFT values at breakpoint boundaries."""
    # Create simple DFT and SFA values with known breakpoints
    dft_x = np.array([0.0, 0.5, -0.5, 1.0])
    sfa_y = np.array([3, 4, 2, 5], dtype=np.int32)  # SFA indices

    # Create breakpoints matching alphabet size 8
    breakpoints = np.array(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        ]
    )

    dist = mindist_dft_sfa_distance(dft_x, sfa_y, breakpoints)

    # DFT values within breakpoint ranges should contribute zero or small distance
    assert dist >= 0


def test_dft_sfa_mindist_dft_outside_range():
    """Test when DFT values fall outside breakpoint ranges."""
    # DFT values far outside typical breakpoint range
    dft_x = np.array([10.0, -10.0, 15.0, -15.0])
    sfa_y = np.array([0, 7, 3, 4], dtype=np.int32)

    breakpoints = np.array(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        ]
    )

    dist = mindist_dft_sfa_distance(dft_x, sfa_y, breakpoints)

    # Should have positive distance due to DFT being outside ranges
    assert dist > 0


def test_dft_sfa_mindist_invalid_shape():
    """Test that invalid input shapes raise errors."""
    dft_x = np.array([[0.0, 1.0]])  # 2D when expecting 1D
    sfa_y = np.array([0, 1], dtype=np.int32)
    breakpoints = np.ones((2, 7))

    with pytest.raises(ValueError):
        mindist_dft_sfa_distance(dft_x, sfa_y, breakpoints)


def test_dft_sfa_mindist_pairwise_self():
    """Test DFT-SFA pairwise distance matrix."""
    x, _ = make_example_3d_numpy(n_cases=8, n_channels=1, n_timepoints=12)

    transform = SFAWhole(word_length=6, alphabet_size=8, norm=True)
    x_sfa, x_dft = transform.fit_transform(x)

    pw = mindist_dft_sfa_pairwise_distance(x_dft, x_sfa, transform.breakpoints)

    # Check shape (DFT to SFA, should be square)
    assert pw.shape == (8, 8)

    # Check non-negativity
    assert np.all(pw >= 0)

    # Diagonal should be zero or near-zero (DFT to corresponding SFA)
    assert np.all(np.diag(pw) < 1e-8)


def test_dft_sfa_mindist_pairwise_manual_comparison():
    """Test pairwise against manual computation."""
    x, _ = make_example_3d_numpy(n_cases=4, n_channels=1, n_timepoints=10)

    transform = SFAWhole(word_length=6, alphabet_size=8, norm=True)
    x_sfa, x_dft = transform.fit_transform(x)

    pw = mindist_dft_sfa_pairwise_distance(x_dft, x_sfa, transform.breakpoints)

    manual_pw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            manual_pw[i, j] = mindist_dft_sfa_distance(
                x_dft[i], x_sfa[j], transform.breakpoints
            )

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_dft_sfa_mindist_single_sample():
    """Test with a single sample."""
    x, _ = make_example_3d_numpy(n_cases=1, n_channels=1, n_timepoints=10)
    y = x + 10

    transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    x_sfa, _ = transform.fit_transform(x)
    _, y_dft = transform.transform(y)

    for i in range(len(x_sfa)):
        dist = mindist_dft_sfa_distance(y_dft[i], x_sfa[i], transform.breakpoints)
        # Normalized series with offset,distance should be 0
        assert dist == 0
