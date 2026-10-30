"""
Tests for NaiveSubsequenceSearch subsequence nearest neighbor search.

We do not test equality for returned indexes due to the unstable nature of argsort
and the fact that the "kind=stable" parameter is not yet supported in numba. We instead
test that the returned index matches the expected distance value.
"""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.similarity_search.subsequence import NaiveSubsequenceSearch
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.utils.numba.general import z_normalise_series_2d


def _reference_distance_profile(X, q, length, normalize):
    """Independent brute-force distance profile of ``q`` over a collection ``X``.

    Squared Euclidean distance (z-normalized per window when ``normalize``), computed
    with pure numpy and no shared kernels. Random data means no constant windows, so no
    constant-window special case is needed.
    """
    n_cases, _, n_timepoints = X.shape
    n_candidates = n_timepoints - length + 1
    profile = np.zeros((n_cases, n_candidates))
    if normalize:
        q = z_normalise_series_2d(q)
    for i in range(n_cases):
        for j in range(n_candidates):
            sub = X[i, :, j : j + length]
            if normalize:
                sub = z_normalise_series_2d(sub)
            profile[i, j] = np.sum((sub - q) ** 2)
    return profile


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_naive_distance_profile_matches_reference(normalize, n_channels):
    """The estimator matches an independent brute-force reference profile.

    Exercises the full ``NaiveSubsequenceSearch.fit``/``compute_distance_profile``
    path -- notably the precomputed ``X_subs_`` (and its z-normalization when
    ``normalize``) -- against an independent numpy reference, for univariate and
    multivariate, normalize True/False.
    """
    length = 8
    X = make_example_3d_numpy(
        n_cases=6,
        n_channels=n_channels,
        n_timepoints=30,
        return_y=False,
    )
    query = make_example_2d_numpy_series(n_channels=n_channels, n_timepoints=length)
    est = NaiveSubsequenceSearch(length=length, normalize=normalize).fit(X)
    got = est.compute_distance_profile(query)
    expected = _reference_distance_profile(X, query, length, normalize)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_naive_matches_mass_distance_profile():
    """The estimator and MASS agree on the distance profile (cross-estimator check)."""
    from aeon.similarity_search.subsequence import MASS

    length = 8
    X = make_example_3d_numpy(n_cases=5, n_channels=2, n_timepoints=30, return_y=False)
    query = make_example_2d_numpy_series(n_channels=2, n_timepoints=length)
    for normalize in (False, True):
        bf = NaiveSubsequenceSearch(length=length, normalize=normalize).fit(X)
        ms = MASS(length=length, normalize=normalize).fit(X)
        np.testing.assert_allclose(
            bf.compute_distance_profile(query),
            ms.compute_distance_profile(query),
            atol=1e-6,
        )


@pytest.mark.parametrize("normalize", [False, True])
def test_naive_predict_finds_self_match(normalize):
    """A query taken from the fitted collection is recovered at distance ~0."""
    length = 10
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=50, return_y=False)
    est = NaiveSubsequenceSearch(length=length, normalize=normalize).fit(X)
    query = X[3, :, 12 : 12 + length].copy()
    indexes, distances = est.predict(query, k=1, allow_trivial_matches=True)
    assert indexes[0, 0] == 3
    assert indexes[0, 1] == 12
    assert_almost_equal(distances[0], 0.0, decimal=5)


@pytest.mark.parametrize("normalize", [False, True])
def test_naive_distance_euclidean_matches_sqrt_of_squared(normalize):
    """``distance="euclidean"`` equals the square root of the default profile."""
    length = 8
    X = make_example_3d_numpy(n_cases=4, n_channels=2, n_timepoints=30, return_y=False)
    query = make_example_2d_numpy_series(n_channels=2, n_timepoints=length)
    squared = (
        NaiveSubsequenceSearch(length=length, normalize=normalize)
        .fit(X)
        .compute_distance_profile(query)
    )
    euclidean = (
        NaiveSubsequenceSearch(length=length, normalize=normalize, distance="euclidean")
        .fit(X)
        .compute_distance_profile(query)
    )
    np.testing.assert_allclose(euclidean, np.sqrt(squared), atol=1e-6)


def test_naive_distance_dtw_matches_reference():
    """``distance="dtw"`` with ``distance_params`` matches direct distance calls."""
    from aeon.distances import dtw_distance

    length = 8
    n_cases, n_timepoints = 3, 20
    X = make_example_3d_numpy(
        n_cases=n_cases, n_channels=2, n_timepoints=n_timepoints, return_y=False
    )
    query = make_example_2d_numpy_series(n_channels=2, n_timepoints=length)
    est = NaiveSubsequenceSearch(
        length=length, distance="dtw", distance_params={"window": 0.2}
    ).fit(X)
    got = est.compute_distance_profile(query)
    n_candidates = n_timepoints - length + 1
    expected = np.zeros((n_cases, n_candidates))
    for i in range(n_cases):
        for j in range(n_candidates):
            expected[i, j] = dtw_distance(X[i, :, j : j + length], query, window=0.2)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_naive_distance_callable():
    """A callable distance is used for the distance profile computation."""

    def _manhattan(x, y):
        return np.sum(np.abs(x - y))

    length = 8
    n_cases, n_timepoints = 3, 20
    X = make_example_3d_numpy(
        n_cases=n_cases, n_channels=1, n_timepoints=n_timepoints, return_y=False
    )
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
    est = NaiveSubsequenceSearch(length=length, distance=_manhattan).fit(X)
    got = est.compute_distance_profile(query)
    n_candidates = n_timepoints - length + 1
    expected = np.zeros((n_cases, n_candidates))
    for i in range(n_cases):
        for j in range(n_candidates):
            expected[i, j] = _manhattan(X[i, :, j : j + length], query)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_naive_invalid_distance_raises():
    """An unknown distance string raises a ValueError at compute time."""
    length = 8
    X = make_example_3d_numpy(n_cases=3, n_channels=1, n_timepoints=20, return_y=False)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
    est = NaiveSubsequenceSearch(length=length, distance="not_a_distance").fit(X)
    with pytest.raises(ValueError):
        est.compute_distance_profile(query)


def test_naive_predict_k_inf_returns_all():
    """``k=np.inf`` returns every admissible candidate for the naive searcher."""
    length = 10
    n_cases, n_timepoints = 3, 40
    X = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=1,
        n_timepoints=n_timepoints,
        return_y=False,
    )
    est = NaiveSubsequenceSearch(length=length).fit(X)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
    indexes, distances = est.predict(query, k=np.inf, allow_trivial_matches=True)
    assert len(indexes) == n_cases * (n_timepoints - length + 1)
    assert len(distances) == len(indexes)
