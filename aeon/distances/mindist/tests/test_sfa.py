"""Tests for SFA mindist distance functions.

This module tests mindist_sfa_distance and mindist_sfa_pairwise_distance,
validating lower bounding and consistency across SFA variants.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist import mindist_sfa_distance, mindist_sfa_pairwise_distance
from aeon.transformations.collection.dictionary_based import SFA, SFAFast, SFAWhole


def test_sfa_mindist_non_negativity():
    """Test that SFA mindist is always non-negative."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    sfa_transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    sfa_transformed, _ = sfa_transform.fit_transform(X)

    for i in range(min(5, X.shape[0])):
        for j in range(i, min(i + 3, X.shape[0])):
            dist = mindist_sfa_distance(
                sfa_transformed[i], sfa_transformed[j], sfa_transform.breakpoints
            )
            assert dist >= 0


def test_sfa_mindist_adjacent_symbols():
    """Test that adjacent SFA symbols contribute zero distance."""
    # Manually create SFA representations
    sfa_x = np.array([0, 1, 2, 3], dtype=np.int32)
    sfa_y = np.array([0, 2, 3, 4], dtype=np.int32)  # Some adjacent differences

    # Create 2D breakpoints for SFA (word_length=4, alphabet_size=8)
    breakpoints = np.array(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        ]
    )

    dist = mindist_sfa_distance(sfa_x, sfa_y, breakpoints)

    # All differences of 1 contribute 0, so total distance is 0
    assert_almost_equal(dist, 0.0, decimal=10)


def test_sfa_mindist_non_adjacent_symbols():
    """Test that non-adjacent SFA symbols contribute non-zero distance."""
    sfa_x = np.array([0, 0, 0, 0], dtype=np.int32)
    sfa_y = np.array([7, 7, 7, 7], dtype=np.int32)  # Large difference

    breakpoints = np.array(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        ]
    )

    dist = mindist_sfa_distance(sfa_x, sfa_y, breakpoints)
    assert dist > 0


def test_sfa_mindist_consistency_across_variants():
    """Test that all SFA variants produce consistent mindist results."""
    X_train, _ = load_unit_test("TRAIN")
    X_train = zscore(X_train.squeeze(), axis=1)
    n = X_train.shape[-1]

    # Use same configuration for all variants
    word_length = 8
    alphabet_size = 8
    histogram_type = "equi-width"

    variants = [
        SFAFast(
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=n,
            binning_method=histogram_type,
            norm=True,
            variance=False,
            lower_bounding_distances=True,
        ),
        SFA(
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=n,
            binning_method=histogram_type,
            norm=True,
            lower_bounding_distances=True,
        ),
        SFAWhole(
            word_length=word_length,
            alphabet_size=alphabet_size,
            binning_method=histogram_type,
            variance=False,
            norm=True,
        ),
    ]

    distances = []
    for sfa in variants:
        sfa.fit(X_train)
        sfa_words, _ = sfa.transform_words(X_train)

        dist = mindist_sfa_distance(sfa_words[0], sfa_words[1], sfa.breakpoints)
        distances.append(dist)

    # All variants should produce the same distance
    assert np.allclose(distances, distances[0])


def test_sfa_mindist_invalid_shape():
    """Test that invalid input shapes raise errors."""
    sfa_x = np.array([[0, 1, 2, 3]])  # 2D when expecting 1D
    sfa_y = np.array([0, 1, 2, 3])
    breakpoints = np.ones((4, 7))

    with pytest.raises(ValueError, match="x and y must be 1D"):
        mindist_sfa_distance(sfa_x, sfa_y, breakpoints)


def test_sfa_mindist_pairwise_self():
    """Test SFA pairwise distance matrix."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)[:10]

    sfa_transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    sfa_transformed, _ = sfa_transform.fit_transform(X)

    pw = mindist_sfa_pairwise_distance(sfa_transformed, None, sfa_transform.breakpoints)

    assert pw.shape == (10, 10)
    assert_almost_equal(np.diag(pw), np.zeros(10), decimal=10)
    assert np.allclose(pw, pw.T)
    assert np.all(pw >= 0)


def test_sfa_mindist_pairwise_manual_comparison():
    """Test pairwise against manual computation."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)[:5]

    sfa_transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    sfa_transformed, _ = sfa_transform.fit_transform(X)

    pw = mindist_sfa_pairwise_distance(sfa_transformed, None, sfa_transform.breakpoints)

    manual_pw = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            manual_pw[i, j] = mindist_sfa_distance(
                sfa_transformed[i], sfa_transformed[j], sfa_transform.breakpoints
            )

    assert_almost_equal(pw, manual_pw, decimal=10)


def test_sfa_mindist_pairwise_multiple_to_multiple():
    """Test pairwise between two different SFA collections."""
    X, _ = load_unit_test("TRAIN")
    Y, _ = load_unit_test("TEST")

    X = zscore(X.squeeze(), axis=1)[:5]
    Y = zscore(Y.squeeze(), axis=1)[:7]

    sfa_transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    sfa_X, _ = sfa_transform.fit_transform(X)
    sfa_Y, _ = sfa_transform.transform(Y)

    pw = mindist_sfa_pairwise_distance(sfa_X, sfa_Y, sfa_transform.breakpoints)

    assert pw.shape == (5, 7)
    assert np.all(pw >= 0)


def test_sfa_mindist_different_word_lengths():
    """Test SFA mindist with different word lengths."""
    X, _ = load_unit_test("TRAIN")
    X = zscore(X.squeeze(), axis=1)

    for word_length in [4, 8, 16]:
        sfa_transform = SFAWhole(word_length=word_length, alphabet_size=8, norm=True)
        sfa_transformed, _ = sfa_transform.fit_transform(X)

        dist = mindist_sfa_distance(
            sfa_transformed[0], sfa_transformed[1], sfa_transform.breakpoints
        )

        assert dist >= 0
