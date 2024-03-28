"""Tests for distance utility function."""

import numpy as np
import pytest

from aeon.distances._shape_dtw import _pad_ts_edges, _transform_subsequences
from aeon.distances._utils import reshape_pairwise_to_multiple

SINGLE_POINT_NOT_SUPPORTED_DISTANCES = ["ddtw", "wddtw", "edr"]


def test_incorrect_input():
    """Test util function incorrect input."""
    x = np.random.rand(10, 2, 2, 10)
    y = np.random.rand(10, 2, 10)
    with pytest.raises(
        ValueError, match="The matrix provided has more than 3 " "dimensions"
    ):
        _make_3d_series(x)
    with pytest.raises(ValueError, match="x and y must be 1D, 2D, or 3D arrays"):
        reshape_pairwise_to_multiple(x, x)
    with pytest.raises(ValueError, match="x and y must be 1D, 2D, or 3D arrays"):
        reshape_pairwise_to_multiple(x, y)


def test_reshape_pairwise_to_multiple():
    """Test function to reshape pairwise distance to multiple distance."""
    x = np.random.rand(5, 2, 10)
    y = np.random.rand(5, 2, 10)
    x2, y2 = reshape_pairwise_to_multiple(x, y)
    assert x2.shape == y2.shape == (5, 2, 10)
    x = np.random.rand(5, 10)
    y = np.random.rand(5, 10)
    x2, y2 = reshape_pairwise_to_multiple(x, y)
    assert x2.shape == y2.shape == (5, 1, 10)
    y = np.random.rand(5)
    assert x2.shape == y2.shape == (5, 1, 10)


def _make_3d_series(x: np.ndarray) -> np.ndarray:
    """Check a series being passed into pairwise is 3d.

    Pairwise assumes it has been passed two sets of series, if passed a single
    series this function reshapes.

    If given a 1d array the time series is reshaped to (1, 1, m). This is so when
    looped over x[i] = (1, m).

    If given a 2d array then the time series is reshaped to (d, 1, m). The dimensions
    are put to the start so the ts can be looped through correctly. When looped over
    the time series x[i] = (1, m).

    Parameters
    ----------
    x: np.ndarray, 2d or 3d

    Returns
    -------
    np.ndarray, 3d
    """
    n_channels = x.ndim
    if n_channels == 1:
        shape = x.shape
        _x = np.reshape(x, (1, 1, shape[0]))
    elif n_channels == 2:
        shape = x.shape
        _x = np.reshape(x, (shape[0], 1, shape[1]))
    elif n_channels > 3:
        raise ValueError(
            "The matrix provided has more than 3 dimensions. This is not"
            "supported. Please provide a matrix with less than "
            "3 dimensions"
        )
    else:
        _x = x
    return _x


def _generate_shape_dtw_params(x: np.ndarray, y: np.ndarray):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    padded_x = _pad_ts_edges(x=x, reach=4)
    padded_y = _pad_ts_edges(x=y, reach=4)

    transformed_x = _transform_subsequences(x=padded_x, reach=4)
    transformed_y = _transform_subsequences(x=padded_y, reach=4)
    return {
        "transformation_precomputed": True,
        "transformed_x": transformed_x,
        "transformed_y": transformed_y,
        "reach": 10,
    }
