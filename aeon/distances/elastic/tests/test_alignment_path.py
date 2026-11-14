"""Tests for alignment paths."""

import numpy as np
import pytest
from numba.typed import List as NumbaList
from numpy.testing import assert_almost_equal

from aeon.distances import alignment_path as compute_alignment_path
from aeon.distances._distance import (
    DISTANCES,
    DISTANCES_DICT,
    SINGLE_POINT_NOT_SUPPORTED_DISTANCES,
)
from aeon.distances.elastic._alignment_paths import derivative_path_to_original_path
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)


def _validate_alignment_path_result(
    x: np.ndarray,
    y: np.ndarray,
    name,
    distance,
    alignment_path,
    check_xy_permuted=True,
):
    original_x = x.copy()
    original_y = y.copy()
    alignment_path_result = alignment_path(x, y)
    callable_alignment_path = DISTANCES_DICT[name]["alignment_path"](x, y)

    assert isinstance(alignment_path_result, tuple)
    assert isinstance(alignment_path_result[0], list)
    assert isinstance(alignment_path_result[1], float)
    assert compute_alignment_path(x, y, method=name) == alignment_path_result
    # Test a callable being passed
    assert callable_alignment_path == alignment_path_result

    distance_result = distance(x, y)
    assert_almost_equal(alignment_path_result[1], distance_result)

    # If unequal length swap where x and y are to ensure it works both ways around
    if original_x.shape[-1] != original_y.shape[-1] and check_xy_permuted:
        _validate_alignment_path_result(
            original_y,
            original_x,
            name,
            distance,
            alignment_path,
            check_xy_permuted=False,
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_alignment_path(dist):
    """Test function to check the alignment path for various distances."""
    if "alignment_path" not in dist:
        return

    # ================== Test equal length ==================
    # Test univariate of shape (n_timepoints,)
    _validate_alignment_path_result(
        make_example_1d_numpy(10, random_state=1),
        make_example_1d_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test univariate of shape (1, n_timepoints)
    _validate_alignment_path_result(
        make_example_2d_numpy_series(10, 1, random_state=1),
        make_example_2d_numpy_series(10, 1, random_state=1),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test multivariate of shape (n_channels, n_timepoints)
    _validate_alignment_path_result(
        make_example_2d_numpy_series(10, 10, random_state=1),
        make_example_2d_numpy_series(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # ================== Test unequal length ==================
    # Test univariate unequal length of shape (n_timepoints,)
    _validate_alignment_path_result(
        make_example_1d_numpy(5, random_state=1),
        make_example_1d_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test univariate unequal length of shape (1, n_timepoints)
    _validate_alignment_path_result(
        make_example_2d_numpy_series(5, 1, random_state=1),
        make_example_2d_numpy_series(10, 1, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test multivariate unequal length of shape (n_channels, n_timepoints)
    _validate_alignment_path_result(
        make_example_2d_numpy_series(5, 10, random_state=1),
        make_example_2d_numpy_series(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # ============== Test single point series ==============
    if (
        dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES
        and dist["name"] != "lcss"
    ):
        # Test single point univariate of shape (1,)
        _validate_alignment_path_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )

        # Test single point univariate of shape (1, 1)
        _validate_alignment_path_result(
            np.array([[10.0]]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )


@pytest.mark.parametrize("name", ["ddtw", "wddtw"])
@pytest.mark.parametrize(
    "n_timepoints, m_timepoints", [(10, 10), (10, 7), (7, 10), (3, 3)]
)
def test_derivative_alignment_path_uses_original_indices(
    name, n_timepoints, m_timepoints
):
    """Test the ddtw and wddtw paths index the original series, not the derivative."""
    x = make_example_2d_numpy_series(n_timepoints, 1, random_state=1)
    y = make_example_2d_numpy_series(m_timepoints, 1, random_state=2)

    path, _ = compute_alignment_path(x, y, method=name)

    assert path[0] == (0, 0)
    assert path[-1] == (n_timepoints - 1, m_timepoints - 1)
    assert {i for i, _ in path} == set(range(n_timepoints))
    assert {j for _, j in path} == set(range(m_timepoints))
    # A warping path moves by at most one point in each series at every step.
    for (i, j), (next_i, next_j) in zip(path, path[1:]):
        assert (next_i - i, next_j - j) in [(1, 1), (1, 0), (0, 1)]


def test_derivative_path_to_original_path():
    """Test the mapping adds one to each index and adds the two corner points."""
    derivative_path = NumbaList([(0, 0), (1, 0), (1, 1), (2, 2)])

    path = derivative_path_to_original_path(derivative_path, 5, 5)

    assert path == [(0, 0), (1, 1), (2, 1), (2, 2), (3, 3), (4, 4)]
