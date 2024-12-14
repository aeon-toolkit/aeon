"""Tests for stomp algorithm."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numba.typed import List
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal

from aeon.distances import get_distance_function
from aeon.similarity_search._commons import get_ith_products
from aeon.similarity_search.matrix_profiles.stomp import (
    _update_dot_products_one_series,
    stomp_normalised_squared_matrix_profile,
    stomp_squared_matrix_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

DATATYPES = ["int64", "float64"]
K_VALUES = [1]


def test__update_dot_products_one_series():
    """Test the _update_dot_product function."""
    X = np.random.rand(1, 50)
    T = np.random.rand(1, 25)
    L = 10
    current_product = get_ith_products(X, T, L, 0)
    for i_query in range(1, T.shape[1] - L + 1):
        new_product = get_ith_products(
            X,
            T,
            L,
            i_query,
        )
        current_product = _update_dot_products_one_series(
            X,
            T,
            current_product,
            L,
            i_query,
        )
        assert_array_almost_equal(new_product, current_product)


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("k", K_VALUES)
def test_stomp_squared_matrix_profile(dtype, k):
    """Test stomp series search."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )

    S = np.asarray([[3, 4, 5, 4, 3, 4, 5, 3, 2, 4, 5]], dtype=dtype)
    L = 3
    mask = np.ones((X.shape[0], X.shape[2] - L + 1), dtype=bool)
    distance = get_distance_function("squared")
    mp, ip = stomp_squared_matrix_profile(X, S, L, mask, k=k)
    for i in range(S.shape[-1] - L + 1):
        q = S[:, i : i + L]

        expected = np.array(
            [
                [distance(q, X[j, :, _i : _i + L]) for _i in range(X.shape[-1] - L + 1)]
                for j in range(X.shape[0])
            ]
        )
        id_bests = np.vstack(
            np.unravel_index(
                np.argsort(expected.ravel(), kind="stable"), expected.shape
            )
        ).T

        for j in range(k):
            assert_almost_equal(mp[i][j], expected[id_bests[j, 0], id_bests[j, 1]])
            assert_equal(ip[i][j], id_bests[j])


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("k", K_VALUES)
def test_stomp_normalised_squared_matrix_profile(dtype, k):
    """Test stomp series search."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )

    S = np.asarray([[3, 4, 5, 4, 3, 4, 5, 3, 2, 4, 5]], dtype=dtype)
    L = 3
    mask = np.ones((X.shape[0], X.shape[2] - L + 1), dtype=bool)
    distance = get_distance_function("squared")
    X_means = []
    X_stds = []

    for i in range(len(X)):
        _mean, _std = sliding_mean_std_one_series(X[i], L, 1)

        X_stds.append(_std)
        X_means.append(_mean)
    X_means = np.asarray(X_means)
    X_stds = np.asarray(X_stds)

    S_means, S_stds = sliding_mean_std_one_series(S, L, 1)

    mp, ip = stomp_normalised_squared_matrix_profile(
        X, S, L, X_means, X_stds, S_means, S_stds, mask, k=k
    )

    for i in range(S.shape[-1] - L + 1):
        q = (S[:, i : i + L] - S_means[:, i]) / S_stds[:, i]

        expected = np.array(
            [
                [
                    distance(
                        q,
                        (X[j, :, _i : _i + L] - X_means[j, :, _i]) / X_stds[j, :, _i],
                    )
                    for _i in range(X.shape[-1] - L + 1)
                ]
                for j in range(X.shape[0])
            ]
        )
        id_bests = np.vstack(
            np.unravel_index(np.argsort(expected.ravel()), expected.shape)
        ).T

        for j in range(k):
            assert_almost_equal(mp[i][j], expected[id_bests[j, 0], id_bests[j, 1]])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_stomp_squared_matrix_profile_unequal_length(dtype):
    """Test stomp with unequal length."""
    X = List(
        [
            np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
            np.array([[1, 2, 4, 4, 5, 6]], dtype=dtype),
        ]
    )
    L = 3
    mask = List(
        [
            np.ones(X[0].shape[1] - L + 1, dtype=bool),
            np.ones(X[1].shape[1] - L + 1, dtype=bool),
        ]
    )
    S = np.asarray([[3, 4, 5, 4, 3, 4, 5, 3, 2, 4, 5]], dtype=dtype)

    distance = get_distance_function("squared")
    mp, ip = stomp_squared_matrix_profile(X, S, L, mask)

    for i in range(S.shape[-1] - L + 1):
        q = S[:, i : i + L]

        expected = [
            [
                distance(q, X[j][:, _i : _i + q.shape[-1]])
                for _i in range(X[j].shape[-1] - q.shape[-1] + 1)
            ]
            for j in range(len(X))
        ]
        assert_almost_equal(mp[i][0], np.concatenate(expected).min())


@pytest.mark.parametrize("dtype", DATATYPES)
@pytest.mark.parametrize("k", K_VALUES)
def test_stomp_squared_matrix_profile_inverse(dtype, k):
    """Test stomp series search for inverse distance."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    S = np.asarray([[3, 4, 5, 4, 3, 4, 5, 3, 2, 4, 5]], dtype=dtype)
    L = 3
    mask = np.ones((X.shape[0], X.shape[2] - L + 1), dtype=bool)
    distance = get_distance_function("squared")
    mp, ip = stomp_squared_matrix_profile(
        X,
        S,
        L,
        mask,
        k=k,
        inverse_distance=True,
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
        expected += 1e-8
        expected = 1 / expected
        id_bests = np.vstack(
            np.unravel_index(np.argsort(expected.ravel()), expected.shape)
        ).T

        for j in range(k):
            assert_almost_equal(mp[i][j], expected[id_bests[j, 0], id_bests[j, 1]])
            assert_equal(ip[i][j], id_bests[j])
