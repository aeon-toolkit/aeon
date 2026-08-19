"""Test _commons.py functions."""

__maintainer__ = ["baraline"]
import numpy as np
import pytest
from numpy.testing import assert_, assert_array_almost_equal

from aeon.similarity_search.subsequence._commons import (
    _extract_top_k_from_dist_profile,
    fft_sliding_dot_product,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
)

K_VALUES = [1, 3, 5]
THRESHOLDS = [np.inf, 1.5]
NN_MATCHES = [False, True]
EXCLUSION_SIZE = [3, 5]


def test_fft_sliding_dot_product():
    """Test the fft_sliding_dot_product function."""
    L = 4
    X = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    Q = make_example_2d_numpy_series(n_channels=1, n_timepoints=L)

    values = fft_sliding_dot_product(X, Q)
    # Compare values[0] only as input is univariate
    assert_array_almost_equal(
        values[0],
        [np.dot(Q[0], X[0, i : i + L]) for i in range(X.shape[1] - L + 1)],
    )


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("allow_nn_matches", NN_MATCHES)
@pytest.mark.parametrize("exclusion_size", EXCLUSION_SIZE)
def test__extract_top_k_from_dist_profile(
    k, threshold, allow_nn_matches, exclusion_size
):
    """Test method to extract the top k candidates from 2D distance profiles."""
    # Create 2D distance profile (n_cases, n_candidates)
    n_cases = 3
    n_candidates = 30
    dist_profile = np.random.rand(n_cases, n_candidates)

    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        dist_profile, k, threshold, allow_nn_matches, exclusion_size
    )

    # Check output shapes
    assert top_k_indexes.ndim == 2
    assert top_k_indexes.shape[1] == 2  # (i_case, i_timestep)
    assert len(top_k_indexes) == len(top_k_distances)

    if len(top_k_indexes) == 0:
        # All distances might be above threshold
        return

    # Verify returned distances match the profile values
    for i in range(len(top_k_indexes)):
        i_case, i_ts = top_k_indexes[i]
        assert_(dist_profile[i_case, i_ts] == top_k_distances[i])

    # All distances should be below threshold
    assert_(np.all(top_k_distances <= threshold))

    # If not allowing trivial matches, check exclusion zones within each case
    if not allow_nn_matches:
        # Group by case and check exclusion
        for case_idx in range(n_cases):
            case_matches = top_k_indexes[top_k_indexes[:, 0] == case_idx, 1]
            if len(case_matches) > 1:
                sorted_ts = np.sort(case_matches)
                assert_(np.all(np.diff(sorted_ts) >= exclusion_size))


def test__extract_top_k_exclusion_zone():
    """Test that exclusion zones correctly prevent neighboring matches."""
    # Create a distance profile with a clear global minimum and neighbors
    n_cases = 2
    n_candidates = 20
    dist_profile = np.ones((n_cases, n_candidates), dtype=np.float64)

    # Place minimum at case 0, position 10
    dist_profile[0, 10] = 0.1
    # Place close neighbors that should be excluded
    dist_profile[0, 9] = 0.2
    dist_profile[0, 11] = 0.2
    dist_profile[0, 8] = 0.25
    dist_profile[0, 12] = 0.25
    # Place a valid second match outside exclusion zone
    dist_profile[0, 5] = 0.3
    # Place a match in another case (should not be affected by exclusion)
    dist_profile[1, 10] = 0.15

    exclusion_size = 3

    # Test with exclusion zones enabled
    top_k_indexes, top_k_distances = _extract_top_k_from_dist_profile(
        dist_profile,
        k=5,
        threshold=np.inf,
        allow_trivial_matches=False,
        exclusion_size=exclusion_size,
    )

    # First match should be at (0, 10) with distance 0.1
    assert top_k_indexes[0, 0] == 0
    assert top_k_indexes[0, 1] == 10
    assert top_k_distances[0] == 0.1

    # Second match should be at (1, 10) with distance 0.15
    # (different case, not affected by exclusion zone)
    assert top_k_indexes[1, 0] == 1
    assert top_k_indexes[1, 1] == 10
    assert top_k_distances[1] == 0.15

    # Third match should be at (0, 5) with distance 0.3
    # (positions 8-12 are excluded in case 0)
    assert top_k_indexes[2, 0] == 0
    assert top_k_indexes[2, 1] == 5
    assert top_k_distances[2] == 0.3

    # Verify no matches within exclusion zone of (0, 10)
    case_0_matches = top_k_indexes[top_k_indexes[:, 0] == 0, 1]
    for ts in case_0_matches:
        if ts != 10:  # Skip the original match
            assert abs(ts - 10) >= exclusion_size

    # Test with exclusion zones disabled - should get neighbors
    top_k_indexes_trivial, top_k_distances_trivial = _extract_top_k_from_dist_profile(
        dist_profile,
        k=5,
        threshold=np.inf,
        allow_trivial_matches=True,
        exclusion_size=exclusion_size,
    )

    # With trivial matches, we should get the neighbors
    assert top_k_distances_trivial[0] == 0.1  # (0, 10)
    assert top_k_distances_trivial[1] == 0.15  # (1, 10)
    # Next should be neighbors at 0.2
    assert top_k_distances_trivial[2] == 0.2
    assert top_k_distances_trivial[3] == 0.2
