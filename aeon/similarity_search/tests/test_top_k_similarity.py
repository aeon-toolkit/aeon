"""Tests for TopKSimilaritySearch."""

__maintainer__ = []

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_array_equal

from aeon.similarity_search.top_k_similarity import TopKSimilaritySearch

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_TopKSimilaritySearch_euclidean(dtype):
    """Test the functionality of TopKSimilaritySearch with Euclidean distance."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = TopKSimilaritySearch(k=1, distance="euclidean")
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = TopKSimilaritySearch(k=3, distance="euclidean")
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    idx = search.predict(q, apply_exclusion_to_result=True)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 4)])

    search = TopKSimilaritySearch(k=1, normalize=True, distance="euclidean")
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])

    idx = search.predict(q, apply_exclusion_to_result=True)
    assert_array_equal(idx, [(1, 2)])

    search = TopKSimilaritySearch(k=1, normalize=True, distance="euclidean")
    search.fit(X)
    idx = search.predict(q, q_index=(1, 2))
    assert_array_equal(idx, [(1, 0)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_TopKSimilaritySearch_custom_func(dtype):
    """Test the functionality of TopKSimilaritySearch using a custom function."""

    def _dist(x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum((x - y) ** 2))

    dist = njit(_dist)
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = TopKSimilaritySearch(k=3, distance=_dist)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    search = TopKSimilaritySearch(k=3, distance=dist)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    search = TopKSimilaritySearch(k=1, normalize=True, distance=dist)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])

    search = TopKSimilaritySearch(k=1, normalize=True, distance=dist)
    search.fit(X)
    idx = search.predict(q, q_index=(1, 2))
    assert_array_equal(idx, [(1, 0)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_TopKSimilaritySearch_change_args(dtype):
    """Test the functionality of TopKSimilaritySearch with different arguments."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = TopKSimilaritySearch(k=1, distance="dtw", distance_args={"window": 0.0})
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = TopKSimilaritySearch(
        k=1, normalize=True, distance="dtw", distance_args={"window": 0.0}
    )
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_TopKSimilaritySearch_speedup(dtype):
    """Test the speedup functionality of TopKSimilaritySearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = TopKSimilaritySearch(
        k=1, distance="euclidean", speed_up="ConvolveDotProduct"
    )
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = TopKSimilaritySearch(
        k=1, distance="euclidean", speed_up="ConvolveDotProduct", normalize=True
    )
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])
