"""Test for bounding matrix."""

import numpy as np
import pytest

from aeon.distances import create_bounding_matrix
from aeon.distances.elastic._bounding_matrix import create_band_bounds
from aeon.distances.elastic._dtw import _dtw_distance, dtw_distance


def test_full_bounding():
    """Test to check the creation of a full bounding matrix."""
    matrix = create_bounding_matrix(10, 10)
    assert np.all(matrix)


def test_window_bounding():
    """Test to check the creation of a windowed bounding matrix."""
    matrix = create_bounding_matrix(10, 10, window=0.2)
    num_true = 0
    num_false = 0
    for row in matrix:
        for val in row:
            if val:
                num_true += 1
            else:
                num_false += 1

    assert num_true == 44
    assert num_false == 56

    unequal_1 = create_bounding_matrix(5, 10, window=0.2)
    unequal_2 = create_bounding_matrix(10, 5, window=0.2).T
    assert np.array_equal(unequal_2, unequal_1)


def test_itakura_parallelogram():
    """Test to check the creation of an Itakura parallelogram bounding matrix."""
    matrix = create_bounding_matrix(10, 10, itakura_max_slope=0.2)
    assert isinstance(matrix, np.ndarray)

    expected_result_5_7 = np.array(
        [
            [True, False, False, False, False, False, False],
            [False, True, True, True, True, False, False],
            [False, False, True, True, True, False, False],
            [False, False, True, True, True, True, False],
            [False, False, False, False, False, False, True],
        ]
    )

    expected_result_7_5 = np.array(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
        ]
    )

    matrix = create_bounding_matrix(5, 7, itakura_max_slope=0.5)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (5, 7)
    assert np.array_equal(matrix, expected_result_5_7)

    matrix = create_bounding_matrix(7, 5, itakura_max_slope=0.5)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (7, 5)
    assert np.array_equal(matrix, expected_result_7_5)


SIZES = [(5, 5), (8, 8), (37, 101), (101, 37), (50, 80), (80, 50), (100, 100)]
WINDOWS = [None, 0.0, 0.05, 0.2, 0.5, 0.9, 1.0]
SLOPES = [0.2, 0.5, 1.0]


@pytest.mark.parametrize("x_size,y_size", SIZES)
def test_band_bounds_match_bounding_matrix(x_size, y_size):
    """create_band_bounds row ranges must equal create_bounding_matrix rows.

    The banded DTW kernels iterate [j_start[i], j_end[i]) instead of testing a
    dense mask, so the bounds must reproduce every dense row exactly, and every
    dense row must be a contiguous run (the kernels assume a single interval per
    row). Itakura cases only run for equal lengths (unsupported otherwise).
    """
    params = [(w, None) for w in WINDOWS]
    if x_size == y_size:
        params += [(None, s) for s in SLOPES]
    for window, slope in params:
        dense = create_bounding_matrix(x_size, y_size, window, slope)
        j_start, j_end = create_band_bounds(x_size, y_size, window, slope)
        for i in range(x_size):
            true_idx = np.flatnonzero(dense[i])
            if len(true_idx) == 0:
                assert j_end[i] <= j_start[i]
                continue
            assert (
                len(true_idx) == true_idx[-1] - true_idx[0] + 1
            ), f"dense row {i} not contiguous for window={window}, slope={slope}"
            assert j_start[i] == true_idx[0] and j_end[i] == true_idx[-1] + 1, (
                f"bounds [{j_start[i]}, {j_end[i]}) != dense "
                f"[{true_idx[0]}, {true_idx[-1] + 1}) at row {i} for "
                f"window={window}, slope={slope}"
            )


@pytest.mark.parametrize("x_size,y_size", SIZES)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_banded_dtw_matches_masked_kernel(x_size, y_size, n_channels):
    """The banded dtw_distance must equal the dense-masked kernel exactly."""
    rng = np.random.RandomState(7)
    x = rng.standard_normal((n_channels, x_size))
    y = rng.standard_normal((n_channels, y_size))
    params = [(w, None) for w in WINDOWS]
    if x_size == y_size:
        params += [(None, s) for s in SLOPES]
    for window, slope in params:
        dense = create_bounding_matrix(x_size, y_size, window, slope)
        expected = _dtw_distance(x, y, dense)
        result = dtw_distance(x, y, window, slope)
        if np.isinf(expected):
            assert np.isinf(result)
        else:
            assert result == pytest.approx(expected, rel=1e-12), (
                f"banded != masked for window={window}, slope={slope}, "
                f"channels={n_channels}, sizes=({x_size}, {y_size})"
            )
