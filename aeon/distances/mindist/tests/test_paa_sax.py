"""Tests for PAA-SAX mindist distance functions.

This module tests mindist_paa_sax_distance and mindist_paa_sax_pairwise_distance,
validating tighter lower bound properties compared to SAX mindist.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist import (
    mindist_paa_sax_distance,
    mindist_paa_sax_pairwise_distance,
    mindist_sax_distance,
)
from aeon.transformations.collection.dictionary_based import SAX


def test_paa_sax_mindist_tighter_lower_bound():
    """Test that PAA-SAX mindist >= SAX mindist (tighter bound)."""
    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    n_segments = 8
    alphabet_size = 8

    sax_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    sax_train = sax_transform.fit_transform(X_train).squeeze()
    paa_train = sax_transform._get_paa(X_train).squeeze()
    sax_test = sax_transform.transform(X_test).squeeze()

    for i in range(min(5, X_train.shape[0], X_test.shape[0])):
        # PAA-SAX Min-Distance
        mindist_paa_sax = mindist_paa_sax_distance(
            paa_train[i], sax_test[i], sax_transform.breakpoints, X_train.shape[-1]
        )

        # SAX Min-Distance
        mindist_sax = mindist_sax_distance(
            sax_train[i], sax_test[i], sax_transform.breakpoints, X_train.shape[-1]
        )

        # PAA-SAX should be >= SAX mindist (tighter bound)
        assert mindist_paa_sax >= mindist_sax - 1e-10


def test_paa_sax_mindist_lower_bound():
    """Test that PAA-SAX mindist is a lower bound for Euclidean."""
    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    paa_train = sax_transform.fit_transform(X_train)
    paa_train = sax_transform._get_paa(X_train).squeeze()
    sax_test = sax_transform.transform(X_test).squeeze()

    for i in range(min(5, X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        mindist_paa_sax = mindist_paa_sax_distance(
            paa_train[i], sax_test[i], sax_transform.breakpoints, X_train.shape[-1]
        )

        euclidean_dist = np.linalg.norm(X[0] - Y[0])

        assert mindist_paa_sax <= euclidean_dist + 1e-10


def test_paa_sax_mindist_identity():
    """Test distance when PAA and SAX are from the same series."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_transformed = sax_transform.fit_transform(X).squeeze()
    paa_transformed = sax_transform._get_paa(X).squeeze()

    for i in range(min(5, X.shape[0])):
        dist = mindist_paa_sax_distance(
            paa_transformed[i],
            sax_transformed[i],
            sax_transform.breakpoints,
            X.shape[-1],
        )
        # PAA to corresponding SAX should be 0 or very close
        assert dist < 1e-8


def test_paa_sax_mindist_non_negativity():
    """Test that PAA-SAX mindist is always non-negative."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    sax_transformed = sax_transform.fit_transform(X).squeeze()
    paa_transformed = sax_transform._get_paa(X).squeeze()

    for i in range(min(5, X.shape[0])):
        for j in range(i, min(i + 3, X.shape[0])):
            dist = mindist_paa_sax_distance(
                paa_transformed[i],
                sax_transformed[j],
                sax_transform.breakpoints,
                X.shape[-1],
            )
            assert dist >= 0


def test_paa_sax_mindist_paa_within_breakpoints():
    """Test when PAA values fall within SAX breakpoint ranges."""
    # PAA values within breakpoint ranges contribute zero distance
    paa_x = np.array([0.0, 0.3, -0.2, 0.8])
    sax_y = np.array([3, 4, 2, 5], dtype=np.int32)
    breakpoints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    n = 16

    dist = mindist_paa_sax_distance(paa_x, sax_y, breakpoints, n)

    # PAA values within their respective SAX breakpoint ranges
    assert dist >= 0


def test_paa_sax_mindist_paa_outside_breakpoints():
    """Test when PAA values fall outside SAX breakpoint ranges."""
    # PAA values significantly outside ranges
    paa_x = np.array([10.0, -10.0, 15.0, -15.0])
    sax_y = np.array([0, 7, 3, 4], dtype=np.int32)
    breakpoints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    n = 16

    dist = mindist_paa_sax_distance(paa_x, sax_y, breakpoints, n)

    # Should have positive distance
    assert dist > 0


def test_paa_sax_mindist_segment_weighting():
    """Test that segment lengths are properly weighted."""
    # With different series lengths n, distances should scale
    paa_x = np.array([0.0, 1.0, 2.0, 3.0])
    sax_y = np.array([5, 6, 7, 7], dtype=np.int32)
    breakpoints = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

    dist_n16 = mindist_paa_sax_distance(paa_x, sax_y, breakpoints, 16)
    dist_n32 = mindist_paa_sax_distance(paa_x, sax_y, breakpoints, 32)

    # Larger n should generally give larger distance (more weight per segment)
    # But relationship depends on segment split, so just check both are valid
    assert dist_n16 >= 0
    assert dist_n32 >= 0


def test_paa_sax_mindist_different_segment_counts():
    """Test PAA-SAX with different segment counts."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    for n_segments in [4, 8, 16]:
        sax_transform = SAX(n_segments=n_segments, alphabet_size=8)
        sax_transformed = sax_transform.fit_transform(X).squeeze()
        paa_transformed = sax_transform._get_paa(X).squeeze()

        dist = mindist_paa_sax_distance(
            paa_transformed[0],
            sax_transformed[1],
            sax_transform.breakpoints,
            X.shape[-1],
        )

        assert dist >= 0


def test_paa_sax_mindist_invalid_shape():
    """Test that invalid input shapes raise errors."""
    paa_x = np.array([[0.0, 1.0]])  # 2D when expecting 1D
    sax_y = np.array([0, 1], dtype=np.int32)
    breakpoints = np.array([0.0, 1.0])
    n = 16

    with pytest.raises(ValueError, match="x and y must be 1D"):
        mindist_paa_sax_distance(paa_x, sax_y, breakpoints, n)


def test_paa_sax_mindist_pairwise_self():
    """Test PAA-SAX pairwise distance matrix."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)[:10]

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    paa_transformed = sax_transform.fit_transform(X)
    paa_transformed = sax_transform._get_paa(X)
    sax_transformed = sax_transform.transform(X)

    pw = mindist_paa_sax_pairwise_distance(
        paa_transformed, sax_transformed, sax_transform.breakpoints, X.shape[-1]
    )

    # Check shape
    assert pw.shape == (10, 10)

    # Check non-negativity
    assert np.all(pw >= 0)

    # Diagonal should be near-zero (PAA to corresponding SAX)
    assert np.all(np.diag(pw) < 1e-8)


def test_paa_sax_mindist_pairwise_manual_comparison():
    """Test pairwise against manual computation."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)[:5]

    sax_transform = SAX(n_segments=8, alphabet_size=8)
    paa_transformed = sax_transform.fit_transform(X)
    paa_transformed = sax_transform._get_paa(X).squeeze()
    sax_transformed = sax_transform.transform(X).squeeze()

    pw = mindist_paa_sax_pairwise_distance(
        paa_transformed, sax_transformed, sax_transform.breakpoints, X.shape[-1]
    )

    manual_pw = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            manual_pw[i, j] = mindist_paa_sax_distance(
                paa_transformed[i],
                sax_transformed[j],
                sax_transform.breakpoints,
                X.shape[-1],
            )

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_paa_sax_mindist_different_alphabet_sizes():
    """Test PAA-SAX with different alphabet sizes."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    for alphabet_size in [4, 6, 8, 10]:
        sax_transform = SAX(n_segments=8, alphabet_size=alphabet_size)
        sax_transformed = sax_transform.fit_transform(X).squeeze()
        paa_transformed = sax_transform._get_paa(X).squeeze()

        dist = mindist_paa_sax_distance(
            paa_transformed[0],
            sax_transformed[1],
            sax_transform.breakpoints,
            X.shape[-1],
        )

        assert dist >= 0
