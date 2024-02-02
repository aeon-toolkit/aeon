"""Tests for distance utility function."""

import numpy as np
import pytest

from aeon.distances._utils import (
    _create_test_distance_numpy,
    _make_3d_series,
    reshape_pairwise_to_multiple,
)


def test_create_test_distance_numpy():
    """Test function to create distance test instance."""
    x = _create_test_distance_numpy(10)
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 10)
    x = _create_test_distance_numpy(10, n_channels=4)
    assert isinstance(x, np.ndarray)
    assert x.shape == (10, 4)
    x = _create_test_distance_numpy(10, n_channels=4, n_timepoints=20)
    assert isinstance(x, np.ndarray)
    assert x.shape == (10, 4, 20)


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
    with pytest.raises(ValueError, match="x and y must be 2D or 3D arrays"):
        reshape_pairwise_to_multiple(x, y)


def test_reshape_pairwise_to_multiple():
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
