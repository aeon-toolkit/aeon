"""Tests for the windowing module."""

import numpy as np
import pytest

from aeon.utils.windowing import (
    _reverse_windowing_iterative,
    _reverse_windowing_strided,
    _reverse_windowing_vectorized,
    reverse_windowing,
    sliding_windows,
)

_DATA = np.random.default_rng(42).integers(0, 10, (10,))
_MULTIVARIATE_DATA = np.random.default_rng(42).integers(0, 10, (10, 2))
_REVERSE_FIXTURES = [
    # ws, stride, padding, results, expected
    (
        1,
        1,
        0,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5, 0.6, 0.5, 0.8, 0.3, 0.4]),
        np.array([0.5, 0.6, 0.5, 0.8, 0.5, 0.6, 0.5, 0.8, 0.3, 0.4]),
    ),
    (
        1,
        2,
        1,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5]),
        np.array([0.5, 0.0, 0.6, 0.0, 0.5, 0.0, 0.8, 0.0, 0.5, 0.0]),
    ),
    (
        1,
        3,
        0,
        np.array([0.5, 0.6, 0.5, 0.8]),
        np.array([0.5, 0.0, 0.0, 0.6, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8]),
    ),
    (
        1,
        4,
        1,
        np.array([0.5, 0.6, 0.5]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.5, 0.0]),
    ),
    (
        1,
        5,
        4,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]),
    ),
    (
        2,
        1,
        0,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5, 0.6, 0.5, 0.8, 0.3]),
        np.array([0.5, 0.55, 0.55, 0.65, 0.65, 0.55, 0.55, 0.65, 0.55, 0.3]),
    ),
    (
        2,
        2,
        0,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5]),
        np.array([0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5]),
    ),
    (
        2,
        3,
        2,
        np.array([0.5, 0.6, 0.5]),
        np.array([0.5, 0.5, 0.0, 0.6, 0.6, 0.0, 0.5, 0.5, 0.0, 0.0]),
    ),
    (
        2,
        4,
        0,
        np.array([0.5, 0.6, 0.5]),
        np.array([0.5, 0.5, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.5, 0.5]),
    ),
    (
        2,
        5,
        3,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0]),
    ),
    (
        3,
        1,
        0,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5, 0.6, 0.5, 0.8]),
        np.array([0.5, 0.55, 0.53, 0.63, 0.6, 0.63, 0.53, 0.63, 0.65, 0.8]),
    ),
    (
        3,
        2,
        1,
        np.array([0.5, 0.6, 0.5, 0.8]),
        np.array([0.5, 0.5, 0.55, 0.6, 0.55, 0.5, 0.65, 0.8, 0.8, 0.0]),
    ),
    (
        3,
        3,
        1,
        np.array([0.5, 0.6, 0.5]),
        np.array([0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.0]),
    ),
    (
        3,
        4,
        3,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0]),
    ),
    (
        3,
        5,
        2,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.6, 0.6, 0.6, 0.0, 0.0]),
    ),
    (
        4,
        1,
        0,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5, 0.6, 0.5]),
        np.array([0.5, 0.55, 0.53, 0.6, 0.6, 0.6, 0.6, 0.53, 0.55, 0.5]),
    ),
    (
        4,
        2,
        0,
        np.array([0.5, 0.6, 0.5, 0.8]),
        np.array([0.5, 0.5, 0.55, 0.55, 0.55, 0.55, 0.65, 0.65, 0.8, 0.8]),
    ),
    (
        4,
        3,
        0,
        np.array([0.5, 0.6, 0.5]),
        np.array([0.5, 0.5, 0.5, 0.55, 0.6, 0.6, 0.55, 0.5, 0.5, 0.5]),
    ),
    (
        4,
        4,
        2,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0]),
    ),
    (
        4,
        5,
        1,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.6, 0.6, 0.6, 0.6, 0.0]),
    ),
    (
        5,
        1,
        0,
        np.array([0.5, 0.6, 0.5, 0.8, 0.5, 0.6]),
        np.array([0.5, 0.55, 0.53, 0.6, 0.58, 0.6, 0.6, 0.63, 0.55, 0.6]),
    ),
    (
        5,
        2,
        1,
        np.array([0.5, 0.6, 0.5]),
        np.array([0.5, 0.5, 0.55, 0.55, 0.53, 0.55, 0.55, 0.5, 0.5, 0.0]),
    ),
    (
        5,
        3,
        2,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.55, 0.55, 0.6, 0.6, 0.6, 0.0, 0.0]),
    ),
    (
        5,
        4,
        1,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.55, 0.6, 0.6, 0.6, 0.6, 0.0]),
    ),
    (
        5,
        5,
        0,
        np.array([0.5, 0.6]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6]),
    ),
]


def _padding(n: int, ws: int, stride: int) -> int:
    return n - int(np.floor((n - ws) / stride)) * stride - ws


def _n_windows(n: int, ws: int, stride: int) -> int:
    return int(np.floor((n - ws) / stride) + 1)


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_sliding_windows_univariate(ws, stride):
    """Test the sliding window function with univariate data."""
    windows, padding = sliding_windows(_DATA, window_size=ws, stride=stride)
    assert windows.shape[1] == ws
    assert windows.shape[0] == _n_windows(_DATA.shape[0], ws, stride)
    assert padding == _padding(_DATA.shape[0], ws, stride)


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_sliding_windows_multivariate(ws, stride):
    """Test the sliding window function with multivariate data."""
    windows, padding = sliding_windows(
        _MULTIVARIATE_DATA, window_size=ws, stride=stride, axis=0
    )
    assert windows.shape[1] == ws * _MULTIVARIATE_DATA.shape[1]
    assert windows.shape[0] == _n_windows(_MULTIVARIATE_DATA.shape[0], ws, stride)
    assert padding == _padding(_MULTIVARIATE_DATA.shape[0], ws, stride)


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_sliding_windows_multivariate_axis(ws, stride):
    """Test the sliding window function with multivariate data."""
    windows, padding = sliding_windows(
        _MULTIVARIATE_DATA.T, window_size=ws, stride=stride, axis=1
    )
    assert windows.shape[0] == ws * _MULTIVARIATE_DATA.shape[1]
    assert windows.shape[1] == _n_windows(_MULTIVARIATE_DATA.shape[0], ws, stride)
    assert padding == _padding(_MULTIVARIATE_DATA.shape[0], ws, stride)


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("ws, stride, padding, results, expected", _REVERSE_FIXTURES)
def test_reverse_windowing(ws, stride, padding, results, expected, force):
    """Test the reverse windowing function."""
    scores = reverse_windowing(
        results,
        window_size=ws,
        reduction=np.nanmean,
        stride=stride,
        padding_length=padding,
        force_iterative=force,
    )
    assert scores.shape[0] == 10
    np.testing.assert_array_almost_equal(scores, expected, decimal=2)


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize(
    "reduction", [np.nanmean, np.nanmedian, np.nanmax, np.nanmin, np.mean, np.average]
)
def test_reverse_windowing_reductions(reduction, force):
    """Test the reverse windowing function with different reduction functions."""
    scores = reverse_windowing(
        _DATA[:9], window_size=2, reduction=reduction, force_iterative=force
    )
    assert scores.shape[0] == 10


def test_reverse_windowing_custom_reduction():
    """Test the reverse windowing function with a custom reduction function."""

    def reduce(x, axis=0):
        np.nan_to_num(x, copy=True)
        return np.sum(x, axis=axis) - np.min(x, axis=axis)

    scores = reverse_windowing(_DATA[:9], window_size=2, reduction=reduce)
    assert scores.shape[0] == 10


def test_combined():
    """Test sliding window and reverse windowing together."""
    ws = 3
    stride = 2
    windows, padding = sliding_windows(_DATA, window_size=ws, stride=stride)
    results = np.random.default_rng(42).random(windows.shape[0])
    mapped = reverse_windowing(
        results,
        window_size=ws,
        reduction=np.nanmedian,
        stride=stride,
        padding_length=padding,
    )
    assert mapped.shape[0] == _DATA.shape[0]
    assert mapped[2] == np.nanmedian(results[:2])


@pytest.mark.parametrize("ws", [1, 2, 3, 4, 5])
def test_all_reverse_windowing_implementations_equal(ws):
    """Test that all reverse windowing implementations return the same result."""
    data = _DATA[:8]
    res_vec = _reverse_windowing_vectorized(data, window_size=3, reduction=np.nanmedian)
    res_strided = _reverse_windowing_strided(
        data, window_size=3, reduction=np.nanmedian, stride=1, padding_length=0
    )
    res_iter = _reverse_windowing_iterative(data, window_size=3, reduction=np.nanmedian)
    np.testing.assert_array_equal(res_vec, res_strided)
    np.testing.assert_array_equal(res_vec, res_iter)
    np.testing.assert_array_equal(res_strided, res_iter)


def test_wrong_input():
    """Test error raises."""
    with pytest.raises(
        ValueError, match="padding_length must be provided when stride " "is not 1"
    ):
        reverse_windowing(_DATA[:9], window_size=2, stride=2)
