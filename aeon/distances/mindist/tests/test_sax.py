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


def test_sax_mindist_basic_lower_bounding():
    """Test that SAX mindist is a lower bound for Euclidean distance."""
    # Load standard test data
    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    n_segments = 8
    alphabet_size = 8

    # Create SAX transformer
    sax_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    sax_train = sax_transform.fit_transform(X_train).squeeze()
    sax_test = sax_transform.transform(X_test).squeeze()

    # Test lower bounding property for several pairs
    for i in range(min(5, X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        # Compute SAX mindist
        mindist_sax = mindist_sax_distance(
            sax_train[i], sax_test[i], sax_transform.breakpoints, X_train.shape[-1]
        )

        # Compute Euclidean distance
        euclidean_dist = np.linalg.norm(X[0] - Y[0])

        # SAX mindist should be <= Euclidean distance
        assert (
            mindist_sax <= euclidean_dist + 1e-10
        ), f"SAX mindist {mindist_sax} exceeds Euclidean {euclidean_dist}"


def test_sax_mindist_identity():
    """Test that identical SAX representations have zero distance."""
    n_segments = 8
    alphabet_size = 8

    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    sax_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    sax_transformed = sax_transform.fit_transform(X).squeeze()

    # Distance from SAX representation to itself should be 0
    for i in range(min(5, X.shape[0])):
        dist = mindist_sax_distance(
            sax_transformed[i],
            sax_transformed[i],
            sax_transform.breakpoints,
            X.shape[-1],
        )
        assert_almost_equal(dist, 0.0, decimal=10)


def test_sax_mindist_symmetry():
    """Test that SAX mindist is symmetric."""
    X_train, _ = load_unit_test("TRAIN")
    X_train = zscore(X_train.squeeze(), axis=1)

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_transformed = sax_transform.fit_transform(X_train).squeeze()

    # Test symmetry for a few pairs
    for i in range(min(3, X_train.shape[0])):
        for j in range(i + 1, min(i + 3, X_train.shape[0])):
            dist_ij = mindist_sax_distance(
                sax_transformed[i],
                sax_transformed[j],
                sax_transform.breakpoints,
                X_train.shape[-1],
            )
            dist_ji = mindist_sax_distance(
                sax_transformed[j],
                sax_transformed[i],
                sax_transform.breakpoints,
                X_train.shape[-1],
            )
            assert_almost_equal(dist_ij, dist_ji, decimal=10)


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
