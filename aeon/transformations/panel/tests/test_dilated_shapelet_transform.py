# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:59:04 2023

@author: antoi
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from aeon.datasets import load_basic_motions
from aeon.transformations.panel.dilated_shapelet_transform import (
    RandomDilatedShapeletTransform,
    compute_normalized_shapelet_dist_vector,
    compute_shapelet_dist_vector,
    compute_shapelet_features,
    compute_shapelet_features_normalized,
    sliding_mean_std_one_series,
)
from aeon.utils.numba.stats import is_prime

DATATYPES = ["int64", "float64", "int32", "float32"]


def test_shapelet_prime_dilation():
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 3, replace=False)
    rdst = RandomDilatedShapeletTransform(
        max_shapelets=10, use_prime_dilations=True
    ).fit(X_train[indices], y_train[indices])
    dilations = rdst.shapelets_[2]
    assert np.all([d == 1 or is_prime(d) for d in dilations])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_features(dtype):
    X = np.asarray([[1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]], dtype=dtype)
    values = np.asarray([[1, 1, 2]], dtype=dtype)
    length = 3
    dilation = 1
    threshold = 0.01

    _min, _argmin, SO = compute_shapelet_features(
        X, values, length, dilation, threshold
    )

    # On some occasion, float32 precision with fasmath retruns things like
    # 2.1835059227370834e-07 instead of 0
    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 0.0
    assert SO == 3.0

    dilation = 2
    threshold = 0.1

    _min, _argmin, SO = compute_shapelet_features(
        X, values, length, dilation, threshold
    )

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 7.0
    assert SO == 1.0

    dilation = 4
    threshold = 2

    _min, _argmin, SO = compute_shapelet_features(
        X, values, length, dilation, threshold
    )

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 3.0
    assert SO == 3.0


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_features_normalized(dtype):
    X = np.asarray(
        [[0, 3 * 1, 1 * 1, 2 * 1, 0, 3 * 2, 1 * 2, 2 * 2, 0, 3 * 3, 1 * 3, 2 * 3]],
        dtype=dtype,
    )
    values = np.asarray([[3 * 4, 1 * 4, 2 * 4]], dtype=dtype)
    length = 3
    dilation = 1
    threshold = 0.01

    X_means, X_stds = sliding_mean_std_one_series(X, length, dilation)
    means = values.mean(axis=1)
    stds = values.std(axis=1)

    _min, _argmin, SO = compute_shapelet_features_normalized(
        X, values, length, dilation, threshold, X_means, X_stds, means, stds
    )
    # On some occasion, float32 precision with fasmath retruns things like
    # 2.1835059227370834e-07 instead of 0
    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 1.0
    assert SO == 3.0

    values = np.asarray([[5 * 4, 5 * 5, 5 * 6]], dtype=dtype)
    dilation = 4
    means = values.mean(axis=1)
    stds = values.std(axis=1)

    _min, _argmin, SO = compute_shapelet_features_normalized(
        X, values, length, dilation, threshold, X_means, X_stds, means, stds
    )

    assert_almost_equal(_min, 0.0, decimal=4)
    # Scale invariance should match with the sets of 3*
    assert _argmin == 1.0
    # And should also do so for the 1* and 2*, all spaced by dilation 4
    assert SO == 3.0


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_normalized_shapelet_dist_vector(dtype):
    # Constant case is tested with dtype int
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            X = (np.random.rand(3, 50)).astype(dtype)
            values = np.random.rand(3, length).astype(dtype)
            d_vect = compute_normalized_shapelet_dist_vector(
                X, values, length, dilation, values.mean(axis=1), values.std(axis=1)
            )
            norm_values = values - values.mean(axis=1, keepdims=True)
            for i_channel in range(X.shape[0]):
                if values[i_channel].std() > 0:
                    norm_values[i_channel] /= values[i_channel].std()
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                for i_channel in range(X.shape[0]):
                    norm_sub = X[i_channel, _idx]
                    norm_sub = norm_sub - norm_sub.mean()
                    if norm_sub.std() > 0:
                        norm_sub /= norm_sub.std()
                    true_vect[i_sub] += ((norm_values[i_channel] - norm_sub) ** 2).sum()
            if dtype == "float32":
                # Fastmath with float32 can sometime produce different of approx 0.0005
                # Any way to compensate this ?
                assert_array_almost_equal(d_vect, true_vect, decimal=3)
            else:
                assert_array_almost_equal(d_vect, true_vect)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_dist_vector(dtype):
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = np.random.rand(3, length).astype(dtype)
            d_vect = compute_shapelet_dist_vector(X, values, length, dilation)
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                for i_channel in range(X.shape[0]):
                    _sub = X[i_channel, _idx]
                    true_vect[i_sub] += ((values[i_channel] - _sub) ** 2).sum()
            assert_array_almost_equal(d_vect, true_vect)
