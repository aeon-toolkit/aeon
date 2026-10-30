"""
Tests for NaiveSeriesSearch whole series nearest neighbor search.

We test that the returned indexes match the expected distance values.
"""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.similarity_search.whole_series import NaiveSeriesSearch
from aeon.testing.data_generation import make_example_3d_numpy


def test_naive_wss_distance_profile_values():
    """Test the default squared Euclidean distance profile values."""
    # Create a small collection of series
    X_collection = np.array(
        [
            [[1.0, 2.0, 3.0]],  # Series 0
            [[4.0, 5.0, 6.0]],  # Series 1
            [[1.0, 2.0, 3.0]],  # Series 2 (same as 0)
        ]
    )
    Q = np.array([[1.0, 2.0, 3.0]])  # Query (same as series 0 and 2)

    dist_profile = NaiveSeriesSearch().fit(X_collection).compute_distance_profile(Q)

    # Distance to series 0 and 2 should be 0 (identical)
    assert_almost_equal(dist_profile[0], 0.0)
    assert_almost_equal(dist_profile[2], 0.0)
    # Distance to series 1: (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
    assert_almost_equal(dist_profile[1], 27.0)


def test_naive_wss_fit_predict():
    """Test NaiveSeriesSearch fit and predict workflow."""
    # Create test data
    n_cases = 10
    n_channels = 1
    n_timepoints = 20
    X = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        return_y=False,
    )

    # Fit the searcher
    searcher = NaiveSeriesSearch(normalize=False, n_jobs=1)
    searcher.fit(X)

    # Query with one of the fitted series (should find itself or very similar)
    query = X[0]
    indexes, distances = searcher.predict(query, k=3)

    # First match should be series 0 with distance 0
    assert indexes[0] == 0
    assert_almost_equal(distances[0], 0.0)


def test_naive_wss_exclude_self():
    """Test NaiveSeriesSearch with X_index to exclude self-match."""
    X = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0]],
        ]
    )

    searcher = NaiveSeriesSearch(normalize=False)
    searcher.fit(X)

    # Query with series 0, excluding it from results
    query = X[0]
    indexes, distances = searcher.predict(query, k=2, X_index=0)

    # Series 0 should not be in results
    assert 0 not in indexes


def test_naive_wss_normalize():
    """Test NaiveSeriesSearch with normalization."""
    X = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[10.0, 20.0, 30.0]],  # Same shape, different scale
        ]
    )

    searcher = NaiveSeriesSearch(normalize=True)
    searcher.fit(X)

    # With normalization, series 0 and 1 should have distance ~0
    # because they have the same shape (just scaled)
    query = X[0]
    indexes, distances = searcher.predict(query, k=2)

    # Both should have very small distances after normalization
    assert_almost_equal(distances[0], 0.0, decimal=5)
    assert_almost_equal(distances[1], 0.0, decimal=5)


def test_naive_wss_k_inf_returns_all():
    """``k=np.inf`` returns all series for the whole-series naive searcher."""
    n_cases = 8
    X = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=1,
        n_timepoints=30,
        return_y=False,
    )
    searcher = NaiveSeriesSearch(normalize=False).fit(X)
    indexes, distances = searcher.predict(X[0], k=np.inf)
    assert len(indexes) == n_cases
    assert len(distances) == n_cases
    # The self-match must be present at distance 0.
    assert 0 in indexes


@pytest.mark.parametrize("normalize", [False, True])
def test_naive_wss_distance_euclidean_matches_sqrt_of_squared(normalize):
    """``distance="euclidean"`` equals the square root of the default profile."""
    X = make_example_3d_numpy(n_cases=6, n_channels=2, n_timepoints=30, return_y=False)
    query = X[0]
    squared = (
        NaiveSeriesSearch(normalize=normalize).fit(X).compute_distance_profile(query)
    )
    euclidean = (
        NaiveSeriesSearch(normalize=normalize, distance="euclidean")
        .fit(X)
        .compute_distance_profile(query)
    )
    np.testing.assert_allclose(euclidean, np.sqrt(squared), atol=1e-6)


def test_naive_wss_distance_dtw_matches_reference():
    """``distance="dtw"`` with ``distance_params`` matches direct distance calls."""
    from aeon.distances import dtw_distance

    n_cases = 5
    X = make_example_3d_numpy(
        n_cases=n_cases, n_channels=2, n_timepoints=20, return_y=False
    )
    query = X[0]
    searcher = NaiveSeriesSearch(distance="dtw", distance_params={"window": 0.2})
    got = searcher.fit(X).compute_distance_profile(query)
    expected = np.array([dtw_distance(X[i], query, window=0.2) for i in range(n_cases)])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_naive_wss_distance_callable():
    """A callable distance is used for the distance profile computation."""

    def _manhattan(x, y):
        return np.sum(np.abs(x - y))

    n_cases = 5
    X = make_example_3d_numpy(
        n_cases=n_cases, n_channels=1, n_timepoints=20, return_y=False
    )
    query = X[0]
    searcher = NaiveSeriesSearch(distance=_manhattan)
    got = searcher.fit(X).compute_distance_profile(query)
    expected = np.array([_manhattan(X[i], query) for i in range(n_cases)])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_naive_wss_invalid_distance_raises():
    """An unknown distance string raises a ValueError at compute time."""
    X = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=20, return_y=False)
    searcher = NaiveSeriesSearch(distance="not_a_distance").fit(X)
    with pytest.raises(ValueError):
        searcher.compute_distance_profile(X[0])


@pytest.mark.parametrize("k", [0, -1, 2.5])
def test_naive_wss_invalid_k_raises(k):
    """Invalid k (non-positive / non-int) raises a clear ValueError."""
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=30, return_y=False)
    searcher = NaiveSeriesSearch().fit(X)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        searcher.predict(X[0], k=k)
