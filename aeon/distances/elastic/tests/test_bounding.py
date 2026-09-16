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


def _valid_bounding_params(x_size, y_size):
    """Return bounding parameters supported by both dense and banded paths."""
    params = [(w, None) for w in WINDOWS]
    if x_size == y_size:
        params += [(None, s) for s in SLOPES]
    return params


@pytest.mark.parametrize("x_size,y_size", SIZES)
def test_band_bounds_match_bounding_matrix(x_size, y_size):
    """Test that band bounds exactly encode the dense bounding matrix.

    ``create_band_bounds`` replaces a dense boolean mask with one half-open column
    interval per row, ``[j_start[i], j_end[i])``. This test verifies that those
    intervals reconstruct the dense mask exactly for full, Sakoe-Chiba, and valid
    Itakura bounds, and that the row bounds satisfy the monotonic invariant relied
    on by the rolling-buffer DTW kernel.
    """
    for window, slope in _valid_bounding_params(x_size, y_size):
        dense = create_bounding_matrix(x_size, y_size, window, slope)
        j_start, j_end = create_band_bounds(x_size, y_size, window, slope)

        reconstructed = np.zeros_like(dense)
        for i, (start, end) in enumerate(zip(j_start, j_end)):
            assert 0 <= start <= end <= y_size, (
                f"invalid bounds [{start}, {end}) at row {i} for "
                f"window={window}, slope={slope}"
            )
            reconstructed[i, start:end] = True

        assert np.array_equal(reconstructed, dense), (
            f"band bounds do not reconstruct dense mask for "
            f"sizes=({x_size}, {y_size}), window={window}, slope={slope}"
        )
        assert np.all(np.diff(j_start) >= 0)
        assert np.all(np.diff(j_end) >= 0)


@pytest.mark.parametrize("x_size,y_size", SIZES)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_banded_dtw_matches_masked_kernel(x_size, y_size, n_channels):
    """Test that public banded DTW returns the dense-mask DTW distance.

    ``dtw_distance`` now computes band bounds directly instead of materializing a
    dense bounding matrix. This compares that public path with the reference
    ``_dtw_distance`` kernel run on the explicit dense mask, for univariate and
    multivariate series, unequal lengths, full windows, Sakoe-Chiba windows, and
    valid Itakura bounds.
    """
    rng = np.random.RandomState(7)
    x = rng.standard_normal((n_channels, x_size))
    y = rng.standard_normal((n_channels, y_size))
    for window, slope in _valid_bounding_params(x_size, y_size):
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


@pytest.mark.parametrize("x_size,y_size", [(8, 40), (40, 8), (15, 30), (30, 15)])
@pytest.mark.parametrize("slope", [0.2, 0.5, 1.0])
def test_itakura_unequal_length_dtw_matches_cost_matrix(x_size, y_size, slope):
    """Test that unequal-length series with Itakura slope match the cost matrix."""
    from aeon.distances import dtw_alignment_path, dtw_cost_matrix

    rng = np.random.default_rng(0)
    x = rng.normal(size=x_size)
    y = rng.normal(size=y_size)

    dist = dtw_distance(x, y, itakura_max_slope=slope)
    cost = dtw_cost_matrix(x, y, itakura_max_slope=slope)[-1, -1]
    path_dist = dtw_alignment_path(x, y, itakura_max_slope=slope)[1]

    assert dist == pytest.approx(cost, rel=1e-12)
    assert dist == pytest.approx(path_dist, rel=1e-12)


def test_itakura_short_series_zero_division_error():
    """Test that series with length < 3 raises ZeroDivisionError consistently."""
    from aeon.distances import dtw_cost_matrix

    rng = np.random.default_rng(0)
    x_short = rng.normal(size=2)
    y_normal = rng.normal(size=10)

    with pytest.raises(ZeroDivisionError):
        dtw_distance(x_short, y_normal, itakura_max_slope=0.5)

    with pytest.raises(ZeroDivisionError):
        dtw_cost_matrix(x_short, y_normal, itakura_max_slope=0.5)
