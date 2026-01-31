"""
Tests for BruteForce whole series nearest neighbor search.

We test that the returned indexes match the expected distance values.
"""

__maintainer__ = ["baraline"]

import numpy as np
from numpy.testing import assert_almost_equal

from aeon.similarity_search.whole_series._dummy import (
    BruteForce,
    _pairwise_squared_distance,
)
from aeon.testing.data_generation import make_example_3d_numpy


def test__pairwise_squared_distance():
    """Test pairwise squared Euclidean distance computation."""
    # Create a small collection of series
    X_collection = np.array(
        [
            [[1.0, 2.0, 3.0]],  # Series 0
            [[4.0, 5.0, 6.0]],  # Series 1
            [[1.0, 2.0, 3.0]],  # Series 2 (same as 0)
        ]
    )
    Q = np.array([[1.0, 2.0, 3.0]])  # Query (same as series 0 and 2)

    dist_profile = _pairwise_squared_distance(X_collection, Q)

    # Distance to series 0 and 2 should be 0 (identical)
    assert_almost_equal(dist_profile[0], 0.0)
    assert_almost_equal(dist_profile[2], 0.0)
    # Distance to series 1: (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
    assert_almost_equal(dist_profile[1], 27.0)


def test_dummy_wss_fit_predict():
    """Test BruteForce fit and predict workflow."""
    # Create test data
    n_cases = 10
    n_channels = 1
    n_timepoints = 20
    X = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=42,
        return_y=False,
    )

    # Fit the searcher
    searcher = BruteForce(normalize=False, n_jobs=1)
    searcher.fit(X)

    # Query with one of the fitted series (should find itself or very similar)
    query = X[0]
    indexes, distances = searcher.predict(query, k=3)

    # First match should be series 0 with distance 0
    assert indexes[0] == 0
    assert_almost_equal(distances[0], 0.0)


def test_dummy_wss_exclude_self():
    """Test BruteForce with X_index to exclude self-match."""
    X = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0]],
        ]
    )

    searcher = BruteForce(normalize=False)
    searcher.fit(X)

    # Query with series 0, excluding it from results
    query = X[0]
    indexes, distances = searcher.predict(query, k=2, X_index=0)

    # Series 0 should not be in results
    assert 0 not in indexes


def test_dummy_wss_normalize():
    """Test BruteForce with normalization."""
    X = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[10.0, 20.0, 30.0]],  # Same shape, different scale
        ]
    )

    searcher = BruteForce(normalize=True)
    searcher.fit(X)

    # With normalization, series 0 and 1 should have distance ~0
    # because they have the same shape (just scaled)
    query = X[0]
    indexes, distances = searcher.predict(query, k=2)

    # Both should have very small distances after normalization
    assert_almost_equal(distances[0], 0.0, decimal=5)
    assert_almost_equal(distances[1], 0.0, decimal=5)
