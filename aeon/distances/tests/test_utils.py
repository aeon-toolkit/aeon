"""Tests for distance utility function."""

import numpy as np
import pytest

from aeon.distances._utils import (
    _make_3d_series,
    _reshape_pairwise_single,
    reshape_pairwise_to_multiple,
)


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


def test_reshape_pairwise_single():
    x = np.random.rand(5, 10)
    y = np.random.rand(5, 10)
    x2, y2 = _reshape_pairwise_single(x, y)
    assert x2.shape == y2.shape == (5, 10)
    x = np.random.rand(10)
    y = np.random.rand(10)
    x2, y2 = _reshape_pairwise_single(x, y)
    assert x2.shape == y2.shape == (1, 10)
    x = np.random.rand(5, 10)
    y = np.random.rand(10)
    x2, y2 = _reshape_pairwise_single(x, y)
    assert x2.ndim == y2.ndim == 2
    assert x2.shape == (5, 10)
    assert y2.shape == (1, 10)
    y2, x2 = _reshape_pairwise_single(y, x)
    assert x2.ndim == y2.ndim == 2
    assert x2.shape == (5, 10)
    assert y2.shape == (1, 10)
