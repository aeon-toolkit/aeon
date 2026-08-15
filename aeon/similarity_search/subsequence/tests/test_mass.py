"""Tests for MASS algorithm."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.similarity_search.subsequence import MASS
from aeon.similarity_search.subsequence._commons import fft_sliding_dot_product
from aeon.similarity_search.subsequence._mass import (
    _normalized_squared_distance_profile,
    _sliding_sum_of_squares,
    _squared_distance_profile,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.utils.numba.general import sliding_mean_std_one_series, z_normalise_series_2d


def _reference_distance_profile(X, q, length, normalize):
    """Independent brute-force distance profile of ``q`` over a collection ``X``.

    Squared Euclidean distance (z-normalized per window when ``normalize``), computed
    with pure numpy and no shared kernels, to check the MASS estimator end to end.
    Assumes no constant windows (random data), so no constant-window special case is
    needed.
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


def test__squared_distance_profile():
    """Test squared distance profile from cached sliding sum-of-squares."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = fft_sliding_dot_product(X, Q)
    # T_ssq is the sliding window sum-of-squares of X, precomputed at fit time.
    T_ssq = _sliding_sum_of_squares(X[np.newaxis], L)[0]
    dist_profile = _squared_distance_profile(QX, T_ssq, Q)
    for i_t in range(X.shape[1] - L + 1):
        assert_almost_equal(dist_profile[i_t], np.sum((X[:, i_t : i_t + L] - Q) ** 2))


def test__normalized_squared_distance_profile():
    """Test Euclidean distance."""
    L = 3
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)
    QX = fft_sliding_dot_product(X, Q)
    X_mean, X_std = sliding_mean_std_one_series(X, L, 1)
    Q_mean = Q.mean(axis=1)
    Q_std = Q.std(axis=1)

    dist_profile = _normalized_squared_distance_profile(
        QX, X_mean, X_std, Q_mean, Q_std, L
    )
    Q = z_normalise_series_2d(Q)
    for i_t in range(X.shape[1] - L + 1):
        S = z_normalise_series_2d(X[:, i_t : i_t + L])
        assert_almost_equal(dist_profile[i_t], np.sum((S - Q) ** 2))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_mass_distance_profile_matches_reference(normalize, n_channels):
    """MASS estimator matches a brute-force reference.

    Exercises the full ``MASS.fit``/``compute_distance_profile`` path -- including the
    cached FFT spectra, the cached sliding sum-of-squares and the
    ``X_means_``/``X_stds_`` normalized wiring -- against an independent numpy
    reference, for univariate and multivariate, normalize True/False.
    """
    length = 8
    X = make_example_3d_numpy(
        n_cases=6,
        n_channels=n_channels,
        n_timepoints=30,
        return_y=False,
    )
    query = make_example_2d_numpy_series(n_channels=n_channels, n_timepoints=length)
    est = MASS(length=length, normalize=normalize).fit(X)
    got = est.compute_distance_profile(query)
    expected = _reference_distance_profile(X, query, length, normalize)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("normalize", [False, True])
def test_mass_predict_finds_self_match(normalize):
    """A query taken from the fitted collection is recovered at distance ~0."""
    length = 10
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=50, return_y=False)
    est = MASS(length=length, normalize=normalize).fit(X)
    query = X[2, :, 15 : 15 + length].copy()
    indexes, distances = est.predict(query, k=1, allow_trivial_matches=True)
    assert indexes[0, 0] == 2
    assert indexes[0, 1] == 15
    assert_almost_equal(distances[0], 0.0, decimal=5)


def test_mass_predict_k_inf_returns_all():
    """``k=np.inf`` returns every admissible candidate for MASS."""
    length = 10
    n_cases, n_timepoints = 3, 40
    X = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=1,
        n_timepoints=n_timepoints,
        return_y=False,
    )
    est = MASS(length=length).fit(X)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
    indexes, distances = est.predict(query, k=np.inf, allow_trivial_matches=True)
    assert len(indexes) == n_cases * (n_timepoints - length + 1)
    assert len(distances) == len(indexes)
