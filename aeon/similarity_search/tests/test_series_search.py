"""Tests for SeriesSearch similarity search algorithm."""

__maintainer__ = ["baraline"]


import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from aeon.similarity_search.series_search import SeriesSearch

DATATYPES = ["int64", "float64"]
K_VALUES = [1, 3]
normalise = [True, False]

# See #2236
# @pytest.mark.parametrize("k", K_VALUES)
# @pytest.mark.parametrize("normalise", normalise)
# def test_SeriesSearch_k(k, normalise):
#    """Test the k and threshold combination of SeriesSearch."""
#    X = np.asarray([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]])
#    S = np.asarray([[3, 4, 5, 4, 3, 4]])
#    L = 3
#
#    search = SeriesSearch(k=k, normalise=normalise)
#    search.fit(X)
#    mp, ip = search.predict(S, L)
#
#    assert mp[0].shape[0] == ip[0].shape[0] == k
#    assert len(mp) == len(ip) == S.shape[1] - L + 1
#    assert ip[0].shape[1] == 2


@pytest.mark.parametrize("dtype", DATATYPES)
def test_SeriesSearch_error_predict(dtype):
    """Test the functionality of SeriesSearch with Euclidean distance."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    S = np.asarray([[3, 4, 5, 4, 3, 4, 5]], dtype=dtype)
    L = 100

    search = SeriesSearch()
    search.fit(X)
    with pytest.raises(ValueError):
        mp, ip = search.predict(S, L)
    L = 3
    S = np.asarray(
        [
            [3, 4, 5, 4, 3, 4],
            [6, 5, 3, 2, 4, 5],
        ],
        dtype=dtype,
    )
    with pytest.raises(ValueError):
        mp, ip = search.predict(S, L)

    S = [6, 5, 3, 2, 4, 5]
    with pytest.raises(TypeError):
        mp, ip = search.predict(S, L)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_SeriesSearch_process_unequal_length(dtype):
    """Test the functionality of SeriesSearch on unequal length data."""
    X = [
        np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
        np.array([[1, 2, 4, 4, 5, 6, 5]], dtype=dtype),
    ]
    S = np.asarray([[3, 4, 5, 4, 3, 4]], dtype=dtype)
    L = 3

    search = SeriesSearch()
    search.fit(X)
    mp, ip = search.predict(S, L)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_SeriesSearch_process_equal_length(dtype):
    """Test the functionality of SeriesSearch on equal length data."""
    X = np.asarray(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [[8, 7, 6, 5, 4, 3, 2, 1]],
        ],
        dtype=dtype,
    )
    S = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype)
    L = 8

    search = SeriesSearch()
    search.fit(X)
    mp, ip = search.predict(S, L)

    assert len(mp) == len(ip) == 1
    assert_array_equal(ip[0], [(0, 0)])
    assert_almost_equal(mp[0][0], 0.0)


# apply_exclusion_throwing_error
# @pytest.mark.parametrize("dtype", DATATYPES)
# @pytest.mark.parametrize("apply_exclusion_to_result", [True, False])
# def test_SeriesSearch_apply_exclusion(dtype, apply_exclusion_to_result):
#     """Test the apply_exclusion_to_result parameter of SeriesSearch."""
#     X = np.asarray(
#         [[[1, 2, 3, 4, 5, 6, 7, 8]], [[2, 3, 4, 5, 6, 7, 8, 9]]], dtype=dtype
#     )
#     S = np.asarray([[3, 4, 5, 6, 7, 8]], dtype=dtype)
#     L = 3

#     search = SeriesSearch()
#     search.fit(X)
#     mp, ip = search.predict(S, L,apply_exclusion_to_result=apply_exclusion_to_result)

#     assert len(mp) == len(ip) == S.shape[1] - L + 1


@pytest.mark.parametrize("dtype", DATATYPES)
def test_SeriesSearch_inverse_distance(dtype):
    """Test the inverse distance parameter of SeriesSearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    S = np.asarray([[3, 4, 5]], dtype=dtype)
    L = 3

    search = SeriesSearch(k=2, inverse_distance=True)
    search.fit(X)
    mp, ip = search.predict(S, L)

    assert_almost_equal(mp[0][:], [0.19245009, 0.28867513])
    assert_array_equal(ip[0][:], [[0, 5], [0, 0]])
