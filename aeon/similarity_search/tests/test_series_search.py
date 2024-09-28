"""Tests for SeriesSearch similarity search algorithm."""

__maintainer__ = ["baraline"]

# See #2071

# import numpy as np
# import pytest
# from numba import njit
#
# from aeon.similarity_search.series_search import SeriesSearch
#
# DATATYPES = ["int64", "float64"]
# K_VALUES = [1, 3]
# NORMALIZE = [True, False]
#
#
# @pytest.mark.parametrize("k", K_VALUES)
# @pytest.mark.parametrize("normalize", NORMALIZE)
# def test_SeriesSearch_naive(k, normalize):
#     """Test the k and threshold combination of SeriesSearch."""
#     X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
#     S = np.asarray([[3, 4, 5, 4, 3, 4]])
#     L = 3
#
#     search = SeriesSearch(k=k, normalize=normalize, speed_up="naive")
#     search.fit(X)
#     mp, ip = search.predict(S, L)
#
#     assert mp[0].shape[0] == ip[0].shape[0] == k
#     assert len(mp) == len(ip) == S.shape[1] - L + 1
#     assert ip[0].shape[1] == 2
#
#
# @pytest.mark.parametrize("k", K_VALUES)
# @pytest.mark.parametrize("normalize", NORMALIZE)
# def test_SeriesSearch_k(k, normalize):
#     """Test the k and threshold combination of SeriesSearch."""
#     X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
#     S = np.asarray([[3, 4, 5, 4, 3, 4]])
#     L = 3
#
#     search = SeriesSearch(k=k, normalize=normalize)
#     search.fit(X)
#     mp, ip = search.predict(S, L)
#
#     assert mp[0].shape[0] == ip[0].shape[0] == k
#     assert len(mp) == len(ip) == S.shape[1] - L + 1
#     assert ip[0].shape[1] == 2
#
#
# @pytest.mark.parametrize("dtype", DATATYPES)
# def test_SeriesSearch_error_predict(dtype):
#     """Test the functionality of SeriesSearch with Euclidean distance."""
#     X = np.asarray(
#         [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
#     )
#     S = np.asarray([[3, 4, 5, 4, 3, 4, 5]], dtype=dtype)
#     L = 100
#
#     search = SeriesSearch()
#     search.fit(X)
#     with pytest.raises(ValueError):
#         mp, ip = search.predict(S, L)
#     L = 3
#     S = np.asarray(
#         [
#             [3, 4, 5, 4, 3, 4],
#             [6, 5, 3, 2, 4, 5],
#         ],
#         dtype=dtype,
#     )
#     with pytest.raises(ValueError):
#         mp, ip = search.predict(S, L)
#
#     S = [6, 5, 3, 2, 4, 5]
#     with pytest.raises(TypeError):
#         mp, ip = search.predict(S, L)
#
#
# @pytest.mark.parametrize("dtype", DATATYPES)
# def test_SeriesSearch_process_unequal_length(dtype):
#     """Test the functionality of SeriesSearch on unequal length data."""
#     X = [
#         np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
#         np.array([[1, 2, 4, 4, 5, 6, 5]], dtype=dtype),
#     ]
#     S = np.asarray([[3, 4, 5, 4, 3, 4]], dtype=dtype)
#     L = 3
#
#     search = SeriesSearch()
#     search.fit(X)
#     mp, ip = search.predict(S, L)
#
#
# @pytest.mark.parametrize("dtype", DATATYPES)
# def test_SeriesSearch_custom_func(dtype):
#     """Test the functionality of SeriesSearch using a custom function."""
#
#     def _dist(x: np.ndarray, y: np.ndarray) -> float:
#         return np.sqrt(np.sum((x - y) ** 2))
#
#     dist = njit(_dist)
#
#     X = np.asarray(
#         [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
#     )
#     S = np.asarray([[3, 4, 5, 4, 3, 4]], dtype=dtype)
#     L = 3
#
#     search = SeriesSearch(distance=dist)
#     search.fit(X)
#     mp, ip = search.predict(S, L)
#
#
# @pytest.mark.parametrize("dtype", DATATYPES)
# def test_SeriesSearch_change_args(dtype):
#     """Test the functionality of SeriesSearch with different arguments."""
#     X = np.asarray(
#         [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
#     )
#     S = np.asarray([[3, 4, 5, 4, 3, 4]], dtype=dtype)
#     L = 3
#
#     search = SeriesSearch(k=1, distance="dtw", distance_args={"window": 0.1})
#     search.fit(X)
#     mp, ip = search.predict(S, L)
