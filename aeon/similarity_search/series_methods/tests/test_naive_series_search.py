"""Tests for naive Euclidean distance profile."""

__maintainer__ = []

import numpy as np
import pytest
from numba.typed import List
from numpy.testing import assert_almost_equal, assert_equal

from aeon.distances import get_distance_function
from aeon.similarity_search.series_methods.naive_series_search import (
    naive_series_search,
)

DATATYPES = ["float64"]
DISTANCES = ["euclidean", "dtw"]
K_VALUES = [1, 3]


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("distance_str", DISTANCES)
@pytest.mark.parametrize("k", K_VALUES)
def test_naive_series_search(dtype, distance_str, k):
    """Test naive series search."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    S = np.asarray([[3, 4, 5, 4, 3, 4, 5, 3, 2, 4, 5]], dtype=dtype)
    L = 3

    distance = get_distance_function(distance_str)
    mp, ip = naive_series_search(
        X, S, L, distance=distance_str, k=k, apply_exclusion_to_result=False
    )

    for i in range(S.shape[-1] - L + 1):
        q = S[:, i : i + L]

        expected = np.array(
            [
                [
                    distance(q, X[j, :, _i : _i + q.shape[-1]])
                    for _i in range(X.shape[-1] - q.shape[-1] + 1)
                ]
                for j in range(X.shape[0])
            ]
        )
        id_bests = np.vstack(
            np.unravel_index(np.argsort(expected.ravel()), expected.shape)
        ).T

        for j in range(k):
            assert_almost_equal(mp[i, j], expected[id_bests[j, 0], id_bests[j, 1]])
            assert_equal(ip[i][j], id_bests[j])


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("distance_str", DISTANCES)
def test_naive_series_search_unequal_length(dtype, distance_str):
    """Test naive distance with unequal length."""
    X = List(
        [
            np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
            np.array([[1, 2, 4, 4, 5, 6]], dtype=dtype),
        ]
    )
    S = np.asarray([[3, 4, 5, 4, 3, 4, 5, 3, 2, 4, 5]], dtype=dtype)
    L = 3
    distance = get_distance_function(distance_str)
    mp, ip = naive_series_search(X, S, L, distance=distance_str)

    for i in range(S.shape[-1] - L + 1):
        q = S[:, i : i + L]

        expected = [
            [
                distance(q, X[j][:, _i : _i + q.shape[-1]])
                for _i in range(X[j].shape[-1] - q.shape[-1] + 1)
            ]
            for j in range(len(X))
        ]
        assert_almost_equal(mp[i, 0], np.concatenate(expected).min())
