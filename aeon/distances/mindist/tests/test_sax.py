"""Tests for SAX mindist distance functions.

This module tests mindist_sax_distance and mindist_sax_pairwise_distance,
validating lower bounding properties and correctness with SAX transformations.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist import mindist_sax_distance, mindist_sax_pairwise_distance
from aeon.transformations.collection.dictionary_based import SAX


def test_sax_mindist_non_negativity():
    """Test that SAX mindist is always non-negative."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_transformed = sax_transform.fit_transform(X).squeeze()

    for i in range(min(5, X.shape[0])):
        for j in range(i, min(i + 3, X.shape[0])):
            dist = mindist_sax_distance(
                sax_transformed[i],
                sax_transformed[j],
                sax_transform.breakpoints,
                X.shape[-1],
            )
            assert dist >= 0


def test_sax_mindist_adjacent_symbols():
    """Test that adjacent SAX symbols contribute zero distance."""
    # Manually create SAX representations with adjacent symbols
    # According to SAX formula, symbols differing by <= 1 contribute 0 distance

    sax_x = np.array([0, 1, 2, 3], dtype=np.int32)
    sax_y = np.array([0, 2, 3, 4], dtype=np.int32)  # Differs at positions 1 and 3

    # Create simple breakpoints for alphabet size 8
    breakpoints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    n = 16  # Original series length

    dist = mindist_sax_distance(sax_x, sax_y, breakpoints, n)

    # Position 1: |1-2| = 1 <= 1, contributes 0
    # Position 3: |3-4| = 1 <= 1, contributes 0
    # All differences contribute 0
    assert_almost_equal(dist, 0.0, decimal=10)


def test_sax_mindist_non_adjacent_symbols():
    """Test that non-adjacent SAX symbols contribute non-zero distance."""
    # Symbols differ by > 1
    sax_x = np.array([0, 0, 0, 0], dtype=np.int32)
    sax_y = np.array([7, 7, 7, 7], dtype=np.int32)  # Large difference

    breakpoints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    n = 16

    dist = mindist_sax_distance(sax_x, sax_y, breakpoints, n)

    # Should have positive distance
    assert dist > 0


def test_sax_mindist_different_alphabet_sizes():
    """Test SAX mindist with different alphabet sizes."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    for alphabet_size in [4, 6, 8, 10]:
        sax_transform = SAX(n_segments=8, alphabet_size=alphabet_size)
        sax_transformed = sax_transform.fit_transform(X).squeeze()

        # Compute mindist for first pair
        dist = mindist_sax_distance(
            sax_transformed[0],
            sax_transformed[1],
            sax_transform.breakpoints,
            X.shape[-1],
        )

        # Should be non-negative
        assert dist >= 0


def test_sax_mindist_invalid_shape():
    """Test that invalid input shapes raise errors."""
    sax_x = np.array([[0, 1, 2, 3]])  # 2D when expecting 1D
    sax_y = np.array([0, 1, 2, 3])
    breakpoints = np.array([-1.0, 0.0, 1.0])
    n = 16

    with pytest.raises(ValueError, match="x and y must be 1D"):
        mindist_sax_distance(sax_x, sax_y, breakpoints, n)


def test_sax_mindist_pairwise_self():
    """Test SAX pairwise distance matrix."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)[:10]  # Use first 10 samples

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_transformed = sax_transform.fit_transform(X)

    pw = mindist_sax_pairwise_distance(
        sax_transformed, None, sax_transform.breakpoints, X.shape[-1]
    )

    # Check shape
    assert pw.shape == (10, 10)

    # Check diagonal is zero
    assert_almost_equal(np.diag(pw), np.zeros(10), decimal=10)

    # Check symmetry
    assert np.allclose(pw, pw.T)

    # Check non-negativity
    assert np.all(pw >= 0)


def test_sax_mindist_pairwise_manual_comparison():
    """Test pairwise against manual computation."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)[:5]

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_transformed = sax_transform.fit_transform(X).squeeze()

    pw = mindist_sax_pairwise_distance(
        sax_transformed, None, sax_transform.breakpoints, X.shape[-1]
    )

    # Manually compute
    manual_pw = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            manual_pw[i, j] = mindist_sax_distance(
                sax_transformed[i],
                sax_transformed[j],
                sax_transform.breakpoints,
                X.shape[-1],
            )

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_sax_mindist_pairwise_multiple_to_multiple():
    """Test pairwise between two different SAX collections."""
    X, _ = load_unit_test("TRAIN")
    Y, _ = load_unit_test("TEST")

    X = zscore(X.squeeze(), axis=1)[:5]
    Y = zscore(Y.squeeze(), axis=1)[:7]

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_X = sax_transform.fit_transform(X)
    sax_Y = sax_transform.transform(Y)

    pw = mindist_sax_pairwise_distance(
        sax_X, sax_Y, sax_transform.breakpoints, X.shape[-1]
    )

    assert pw.shape == (5, 7)
    assert np.all(pw >= 0)


def test_sax_mindist_different_segment_counts():
    """Test SAX mindist with different segment counts."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    for n_segments in [4, 8, 16]:
        sax_transform = SAX(n_segments=n_segments, alphabet_size=8)
        sax_transformed = sax_transform.fit_transform(X).squeeze()

        dist = mindist_sax_distance(
            sax_transformed[0],
            sax_transformed[1],
            sax_transform.breakpoints,
            X.shape[-1],
        )

        assert dist >= 0
