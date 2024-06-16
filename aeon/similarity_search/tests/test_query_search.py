"""Tests for QuerySearch."""

__maintainer__ = []

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_almost_equal, assert_array_equal

from aeon.similarity_search.query_search import QuerySearch

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_mean_std_equal_length(dtype):
    """Test the mean and std computation of QuerySearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(normalize=True)
    search.fit(X)
    _ = search.predict(q, X_index=(1, 2))
    for i in range(len(X)):
        for j in range(X[i].shape[1] - q.shape[1] + 1):
            subsequence = X[i, :, j : j + q.shape[1]]
            assert_almost_equal(search.X_means_[i][:, j], subsequence.mean(axis=-1))
            assert_almost_equal(search.X_stds_[i][:, j], subsequence.std(axis=-1))


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_mean_std_unequal_length(dtype):
    """Test the mean and std computation of QuerySearch on unequal length data."""
    X = [
        np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
        np.array([[1, 2, 4, 4, 5, 6, 5]], dtype=dtype),
    ]

    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(normalize=True)
    search.fit(X)
    _ = search.predict(q, X_index=(1, 2))
    for i in range(len(X)):
        for j in range(X[i].shape[1] - q.shape[1] + 1):
            subsequence = X[i][:, j : j + q.shape[1]]
            assert_almost_equal(search.X_means_[i][:, j], subsequence.mean(axis=-1))
            assert_almost_equal(search.X_stds_[i][:, j], subsequence.std(axis=-1))


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_threshold_and_k(dtype):
    """Test the k and threshold combination of QuerySearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=3, threshold=1)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_inverse_distance(dtype):
    """Test the inverse distance parameter of QuerySearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=1, inverse_distance=True)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 5)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_euclidean(dtype):
    """Test the functionality of QuerySearch with Euclidean distance."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=1)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = QuerySearch(k=3)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    idx = search.predict(q, apply_exclusion_to_result=True)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 4)])

    search = QuerySearch(k=1, normalize=True)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])

    idx = search.predict(q, apply_exclusion_to_result=True)
    assert_array_equal(idx, [(1, 2)])

    search = QuerySearch(k=1, normalize=True)
    search.fit(X)
    idx = search.predict(q, X_index=(1, 2))
    assert_array_equal(idx, [(1, 0)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_euclidean_unequal_length(dtype):
    """Test the functionality of QuerySearch on unequal length data."""
    X = [
        np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
        np.array([[1, 2, 4, 4, 5, 6, 5]], dtype=dtype),
    ]

    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=1)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = QuerySearch(k=3)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    idx = search.predict(q, apply_exclusion_to_result=True)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 4)])

    search = QuerySearch(k=1, normalize=True)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])

    idx = search.predict(q, apply_exclusion_to_result=True)
    assert_array_equal(idx, [(1, 2)])

    search = QuerySearch(k=1, normalize=True)
    search.fit(X)
    idx = search.predict(q, X_index=(1, 2))
    assert_array_equal(idx, [(1, 0)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_custom_func(dtype):
    """Test the functionality of QuerySearch using a custom function."""

    def _dist(x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum((x - y) ** 2))

    dist = njit(_dist)
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=3, distance=_dist)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    search = QuerySearch(k=3, distance=dist)
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2), (1, 2), (1, 1)])

    search = QuerySearch(k=1, normalize=True, distance=dist)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])

    search = QuerySearch(k=1, normalize=True, distance=dist)
    search.fit(X)
    idx = search.predict(q, X_index=(1, 2))
    assert_array_equal(idx, [(1, 0)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_change_args(dtype):
    """Test the functionality of QuerySearch with different arguments."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=1, distance="dtw", distance_args={"window": 0.0})
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = QuerySearch(
        k=1, normalize=True, distance="dtw", distance_args={"window": 0.0}
    )
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_QuerySearch_speedup(dtype):
    """Test the speedup functionality of QuerySearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = QuerySearch(k=1, speed_up="fastest")
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, [(0, 2)])

    search = QuerySearch(
        k=1,
        distance="euclidean",
        speed_up="fastest",
        normalize=True,
    )
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, [(1, 2)])
