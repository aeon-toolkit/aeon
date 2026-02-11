"""Tests for distance factory functions."""

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_almost_equal

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)


@njit(cache=True, fastmath=True, inline="always")
def _fake_univariate_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Fake distance for 1D arrays."""
    dist = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        dist += abs(x[i] - y[i])
    return dist


@njit(cache=True, fastmath=True)
def _fake_distance_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Fake distance for 2D arrays (multivariate)."""
    dist = 0.0
    min_channels = min(x.shape[0], y.shape[0])
    for c in range(min_channels):
        dist += _fake_univariate_distance(x[c], y[c])
    return dist


class TestBuildDistance:
    """Test build_distance factory."""

    def test_1d_input(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        result = distance(x, y)
        expected = 9.0

        assert result == expected
        assert distance.__name__ == "fake_distance"

    def test_2d_input(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        result = distance(x, y)
        expected = 18.0

        assert result == expected

    def test_invalid_dimensions_raises(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )

        x = np.array([[[1.0, 2.0, 3.0]]])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="x must be 1D or 2D"):
            distance(x, y)

    def test_symmetric(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        assert distance(x, y) == distance(y, x)

    def test_zero_when_identical(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert distance(x, x) == 0.0


class TestBuildPairwiseDistance:
    """Test build_pairwise_distance factory."""

    def test_self_distances(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )
        pairwise = build_pairwise_distance(
            core_distance=distance,
            name="fake",
        )

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        result = pairwise(X)

        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0  # Distance to self
        assert result[1, 1] == 0.0
        assert result[2, 2] == 0.0
        assert result[0, 1] == result[1, 0]  # Symmetric

    def test_cross_distances(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )
        pairwise = build_pairwise_distance(
            core_distance=distance,
            name="fake",
        )

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        result = pairwise(X, y)

        assert result.shape == (2, 2)
        assert result[0, 0] == 18.0

    def test_unequal_length_list(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )
        pairwise = build_pairwise_distance(
            core_distance=distance,
            name="fake",
        )

        X = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0, 7.0])]

        result = pairwise(X)

        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0

    def test_multivariate(self):
        distance = build_distance(
            core_distance=_fake_distance_2d,
            name="fake",
        )
        pairwise = build_pairwise_distance(
            core_distance=distance,
            name="fake",
        )

        X = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        )

        result = pairwise(X)

        assert result.shape == (3, 3)
        assert np.all(np.diag(result) == 0.0)


class TestSquaredDistanceIntegration:
    """Integration tests using actual squared_distance exports."""

    def test_squared_pairwise_import(self):
        from aeon.distances.pointwise._squared import squared_pairwise_distance

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = squared_pairwise_distance(X)

        assert result.shape == (2, 2)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0

    def test_squared_matches_expected(self):
        from aeon.distances.pointwise._squared import squared_distance

        x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

        result = squared_distance(x, y)
        expected = 1000.0

        assert result == expected

    def test_squared_pairwise_matches_expected(self):
        from aeon.distances.pointwise._squared import squared_pairwise_distance

        X = np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
        y = np.array([[[11, 12, 13]], [[14, 15, 16]], [[17, 18, 19]]])

        result = squared_pairwise_distance(X, y)

        expected = np.array(
            [
                [300.0, 507.0, 768.0],
                [147.0, 300.0, 507.0],
                [48.0, 147.0, 300.0],
            ]
        )

        assert_almost_equal(result, expected)
