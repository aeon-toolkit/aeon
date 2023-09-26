# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:21:00 2023

@author: antoi
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.distances import euclidean_distance
from aeon.similarity_search.distance_profiles.normalized_naive_euclidean import (
    normalized_naive_euclidean_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalized_naive_euclidean(dtype):
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    Q = np.asarray([[3, 4, 5]], dtype=dtype)

    search_space_size = X.shape[-1] - Q.shape[-1] + 1

    X_means = np.zeros((2, 1, search_space_size))
    X_stds = np.zeros((2, 1, search_space_size))

    for i in range(2):
        _mean, _std = sliding_mean_std_one_series(X[i], Q.shape[-1], 1)
        X_stds[i] = _std
        X_means[i] = _mean

    Q_means = Q.mean(axis=-1)
    Q_stds = Q.std(axis=-1)

    dist_profile = normalized_naive_euclidean_profile(
        X, Q, X_means, X_stds, Q_means, Q_stds
    )

    _Q = (Q - Q_means) / Q_stds
    expected = np.array(
        [
            [
                euclidean_distance(
                    _Q,
                    (X[j, :, i : i + Q.shape[-1]] - X_means[j, :, i]) / X_stds[j, :, i],
                )
                for i in range(X.shape[-1] - Q.shape[-1] + 1)
            ]
            for j in range(X.shape[0])
        ]
    )
    assert_array_almost_equal(dist_profile, expected)
