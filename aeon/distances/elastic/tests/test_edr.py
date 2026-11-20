"""Tests for EDR distance."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import edr_cost_matrix, edr_distance, edr_pairwise_distance


def _edr_reference(x, y, epsilon):
    """Recursive EDR definition: EDR(R, empty) = len(R), normalised by max length."""

    def edr(i, j):
        if i == len(x):
            return len(y) - j
        if j == len(y):
            return len(x) - i
        subcost = 0 if abs(x[i] - y[j]) < epsilon else 1
        return min(edr(i + 1, j + 1) + subcost, edr(i + 1, j) + 1, edr(i, j + 1) + 1)

    return edr(0, 0) / max(len(x), len(y))


def test_edr_dissimilar_series_of_different_length():
    """Nothing in x can match y, so every point is edited and the distance is 1."""
    x = np.zeros((1, 10))
    y = np.array([[100.0]])
    assert edr_distance(x, y, epsilon=0.5) == 1.0
    assert edr_distance(y, x, epsilon=0.5) == 1.0
    assert edr_pairwise_distance(x[np.newaxis], y[np.newaxis], epsilon=0.5)[0, 0] == 1.0


def test_edr_matches_definition():
    """Leading gaps cost the number of deleted points."""
    rng = np.random.default_rng(0)
    for n, m in [(1, 4), (5, 5), (7, 3), (4, 9)]:
        x = rng.normal(size=(1, n))
        y = rng.normal(size=(1, m))
        expected = _edr_reference(x[0], y[0], 0.5)
        assert_allclose(edr_distance(x, y, epsilon=0.5), expected)
        assert_allclose(
            edr_cost_matrix(x, y, epsilon=0.5)[-1, -1] / max(n, m), expected
        )


def test_edr_bounded_path():
    """The bounded implementation (window given) also charges leading gaps."""
    x = np.zeros((1, 10))
    y = np.array([[100.0]])
    with pytest.warns(FutureWarning):
        assert edr_distance(x, y, window=1.0, epsilon=0.5) == 1.0


def test_edr_leading_and_trailing_gaps_cost_the_same():
    """Moving the matching point from the end of x to the start must not matter."""
    x = np.array([[0.0, 0.0, 0.0, 0.0, 5.0]])
    y = np.array([[5.0]])
    assert edr_distance(x, y, epsilon=0.5) == 0.8
    assert edr_distance(x[:, ::-1], y, epsilon=0.5) == 0.8
