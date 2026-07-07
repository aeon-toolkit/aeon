"""
Tests for stomp algorithm.

We do not test equality for returned indexes due to the unstable nature of argsort
and the fact that the "kind=stable" parameter is not yet supported in numba. We instead
test that the returned index match the expected distance value.
"""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from aeon.similarity_search.series._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile,
    get_ith_products,
)
from aeon.similarity_search.series.motifs._stomp import (
    StompMotif,
    _stomp,
    _stomp_normalized,
)
from aeon.similarity_search.series.neighbors._dummy import (
    _naive_squared_distance_profile,
)
from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.utils.numba.general import (
    get_all_subsequences,
    sliding_mean_std_one_series,
    z_normalise_series_3d,
)

MOTIFS_SIZE_VALUES = [1, 3]
THRESHOLD = [np.inf, 0.75]
THRESHOLD_NORM = [np.inf, 4.5]
NN_MATCHES = [True, False]
INVERSE = [True, False]


@pytest.mark.parametrize("motif_size", MOTIFS_SIZE_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLD)
@pytest.mark.parametrize("allow_trivial_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
def test__stomp(motif_size, threshold, allow_trivial_matches, inverse_distance):
    """Test STOMP method."""
    L = 3

    X_A = make_example_2d_numpy_series(
        n_channels=2,
        n_timepoints=10,
    )
    X_B = make_example_2d_numpy_series(n_channels=2, n_timepoints=10)
    AdotB = get_ith_products(X_B, X_A, L, 0)

    exclusion_size = L
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _stomp(
        X_A,
        X_B,
        AdotB,
        L,
        motif_size,
        threshold,
        allow_trivial_matches,
        exclusion_size,
        inverse_distance,
        False,
    )
    # For each query of size L in T
    X_B_subs = get_all_subsequences(X_B, L, 1)
    X_A_subs = get_all_subsequences(X_A, L, 1)
    for i in range(X_A.shape[1] - L + 1):
        dist_profile = _naive_squared_distance_profile(X_B_subs, X_A_subs[i])
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)

        top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
            dist_profile, motif_size, threshold, allow_trivial_matches, exclusion_size
        )
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        assert_array_almost_equal(MP[i], top_k_distances)

        # Check that the index in IP correspond to a distance profile point
        # with value equal to the corresponding MP point.
        for j, index in enumerate(top_k_indexes):
            assert_almost_equal(MP[i][j], dist_profile[index])


@pytest.mark.parametrize("is_self_mp", [True, False])
def test__stomp_self_matrix_profile_exclusion(is_self_mp):
    """Test the self-matrix-profile exclusion-zone branch is exercised."""
    L = 3
    X_A = make_example_2d_numpy_series(n_channels=2, n_timepoints=10)
    AdotB = get_ith_products(X_A, X_A, L, 0)
    exclusion_size = L

    MP, IP = _stomp(
        X_A, X_A, AdotB, L, 1, np.inf, False, exclusion_size, False, is_self_mp
    )
    assert len(MP) == X_A.shape[1] - L + 1


@pytest.mark.parametrize("is_self_mp", [True, False])
def test__stomp_normalized_self_matrix_profile_exclusion(is_self_mp):
    """Test the normalised self-matrix-profile exclusion-zone branch is exercised."""
    L = 3
    X_A = make_example_2d_numpy_series(n_channels=2, n_timepoints=10)
    X_A_means, X_A_stds = sliding_mean_std_one_series(X_A, L, 1)
    AdotB = get_ith_products(X_A, X_A, L, 0)
    exclusion_size = L

    MP, IP = _stomp_normalized(
        X_A,
        X_A,
        AdotB,
        X_A_means,
        X_A_stds,
        X_A_means,
        X_A_stds,
        L,
        1,
        np.inf,
        False,
        exclusion_size,
        False,
        is_self_mp,
    )
    assert len(MP) == X_A.shape[1] - L + 1


@pytest.mark.parametrize("normalize", [True, False])
def test_stomp_motif_fit_predict_self_motifs(normalize):
    """Test StompMotif fit_predict computes self-motifs with k_motifs ranking."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=20)
    motif = StompMotif(length=3, normalize=normalize)

    IP, MP = motif.fit_predict(X, k=2, motif_size=1)

    assert len(IP) == len(MP) == 2


def test_stomp_motif_r_motifs_extraction():
    """Test StompMotif with the r_motifs ranking method."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=20)
    motif = StompMotif(length=3, normalize=False)

    IP, MP = motif.fit_predict(
        X,
        k=2,
        motif_size=np.inf,
        dist_threshold=1.0,
        motif_extraction_method="r_motifs",
    )

    assert len(IP) == len(MP) == 2


def test_stomp_motif_predict_relative_to_other_series():
    """Test StompMotif predict against a series different from the fitted one."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=20)
    Y = make_example_2d_numpy_series(n_channels=1, n_timepoints=15)
    motif = StompMotif(length=3, normalize=True)
    motif.fit(X)

    IP, MP = motif.predict(Y, k=2, motif_size=1, is_self_computation=False)

    assert len(IP) == len(MP) == 2


def test_stomp_motif_invalid_extraction_method_raises():
    """Test an invalid motif_extraction_method raises a ValueError."""
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=20)
    motif = StompMotif(length=3, normalize=False)
    motif.fit(X)

    with pytest.raises(ValueError, match="Expected motif_extraction_method"):
        motif.predict(X, motif_extraction_method="bogus", is_self_computation=True)


def test_stomp_motif_get_test_params():
    """Test the default test parameters are valid and usable."""
    params = StompMotif._get_test_params()
    motif = StompMotif(**params)
    assert isinstance(motif, StompMotif)


def test_stomp_motif_get_test_params_unknown_set_raises():
    """Test an unknown parameter_set raises a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        StompMotif._get_test_params(parameter_set="bogus")


@pytest.mark.parametrize("motif_size", MOTIFS_SIZE_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLD_NORM)
@pytest.mark.parametrize("allow_trivial_matches", NN_MATCHES)
@pytest.mark.parametrize("inverse_distance", INVERSE)
def test__stomp_normalised(
    motif_size, threshold, allow_trivial_matches, inverse_distance
):
    """Test STOMP normalised method."""
    L = 3

    X_A = make_example_2d_numpy_series(
        n_channels=2,
        n_timepoints=10,
    )
    X_B = make_example_2d_numpy_series(n_channels=2, n_timepoints=10)
    X_A_means, X_A_stds = sliding_mean_std_one_series(X_A, L, 1)
    X_B_means, X_B_stds = sliding_mean_std_one_series(X_B, L, 1)
    AdotB = get_ith_products(X_B, X_A, L, 0)

    exclusion_size = L
    # MP : distances to best matches for each query
    # IP : Indexes of best matches for each query
    MP, IP = _stomp_normalized(
        X_A,
        X_B,
        AdotB,
        X_A_means,
        X_A_stds,
        X_B_means,
        X_B_stds,
        L,
        motif_size,
        threshold,
        allow_trivial_matches,
        exclusion_size,
        inverse_distance,
        False,
    )
    # For each query of size L in T
    X_B_subs = z_normalise_series_3d(get_all_subsequences(X_B, L, 1))
    X_A_subs = z_normalise_series_3d(get_all_subsequences(X_A, L, 1))
    for i in range(X_A.shape[1] - L + 1):
        dist_profile = _naive_squared_distance_profile(X_B_subs, X_A_subs[i])
        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        if inverse_distance:
            dist_profile = _inverse_distance_profile(dist_profile)
        top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
            dist_profile, motif_size, threshold, allow_trivial_matches, exclusion_size
        )

        # Check that the top matches extracted have the same value that the
        # top matches in the distance profile
        assert_array_almost_equal(MP[i], top_k_distances)

        # Check that the index in IP correspond to a distance profile point
        # with value equal to the corresponding MP point.
        for j, index in enumerate(top_k_indexes):
            assert_almost_equal(MP[i][j], dist_profile[index])
