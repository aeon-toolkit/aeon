# -*- coding: utf-8 -*-
"""Tests for dilated shapelet transform functions."""

__author__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.distances import manhattan_distance
from aeon.transformations.collection.dilated_shapelet_transform import (
    RandomDilatedShapeletTransform,
    compute_shapelet_dist_vector,
    compute_shapelet_features,
    get_all_subsequences,
    normalize_subsequences,
)
from aeon.utils.numba.stats import is_prime

DATATYPES = ["int64", "float64"]


def test_shapelet_prime_dilation():
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 3, replace=False)
    rdst = RandomDilatedShapeletTransform(
        max_shapelets=10, use_prime_dilations=True
    ).fit(X_train[indices], y_train[indices])
    dilations = rdst.shapelets_[2]
    assert np.all([d == 1 or is_prime(d) for d in dilations])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_normalize_subsequences(dtype):
    X = np.asarray([[[1, 1, 1]], [[1, 1, 1]]], dtype=dtype)
    X_norm = normalize_subsequences(X, X.mean(axis=2), X.std(axis=2))
    assert np.all(X_norm == 0)
    assert np.all(X.shape == X_norm.shape)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_get_all_subsequences(dtype):
    X = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype)
    length = 3
    dilation = 1
    X_subs = get_all_subsequences(X, length, dilation)
    X_true = np.asarray(
        [
            [[1, 2, 3]],
            [[2, 3, 4]],
            [[3, 4, 5]],
            [[4, 5, 6]],
            [[5, 6, 7]],
            [[6, 7, 8]],
        ],
        dtype=dtype,
    )
    assert_array_equal(X_subs, X_true)

    length = 3
    dilation = 2
    X_subs = get_all_subsequences(X, length, dilation)
    X_true = np.asarray(
        [
            [[1, 3, 5]],
            [[2, 4, 6]],
            [[3, 5, 7]],
            [[4, 6, 8]],
        ],
        dtype=dtype,
    )
    assert_array_equal(X_subs, X_true)


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_features(dtype):
    X = np.asarray([[1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]], dtype=dtype)
    values = np.asarray([[1, 1, 2]], dtype=dtype)
    length = 3
    dilation = 1
    threshold = 0.01
    X_subs = get_all_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, length, threshold)

    # On some occasion, float32 precision with fasmath retruns things like
    # 2.1835059227370834e-07 instead of 0
    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 0.0
    assert SO == 3.0

    dilation = 2
    threshold = 0.1
    X_subs = get_all_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, length, threshold)

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 7.0
    assert SO == 1.0

    dilation = 4
    threshold = 2
    X_subs = get_all_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, length, threshold)

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 3.0
    assert SO == 3.0


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_dist_vector(dtype):
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = np.random.rand(3, length).astype(dtype)
            X_subs = get_all_subsequences(X, length, dilation)
            d_vect = compute_shapelet_dist_vector(X_subs, values, length)
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                _sub = X[:, _idx]
                true_vect[i_sub] += manhattan_distance(values, _sub)
            assert_array_almost_equal(d_vect, true_vect)


shapelet_transform_unit_test_data = np.array(
    [
        [0.58048731, 8.0, 1.0, 0.98290187, 10.0, 2.0, 0.0, 1.0, 1.0],
        [0.53932398, 8.0, 1.0, 0.0, 10, 2.0, 0.42051204, 3.0, 0.0],
        [0.0, 8.0, 1.0, 1.3005285, 10.0, 2.0, 0.14676179, 1.0, 1.0],
        [1.06848721, 8.0, 1.0, 6.2313152, 10.0, 1.0, 0.40016587, 3.0, 0.0],
        [1.31181694, 8.0, 1.0, 1.02493714, 10.0, 3.0, 0.11072912, 1.0, 1.0],
    ]
)


def test_rdst_on_unit_test():
    """Test of ShapeletTransform on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomDilatedShapeletTransform(max_shapelets=3, random_state=0)
    st.fit(X_train[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train[indices])
    assert_array_almost_equal(data, shapelet_transform_unit_test_data, decimal=4)


shapelet_transform_basic_motions_data = np.array(
    [
        [26.64112374, 25.0, 4.0, 96.47472839, 5.0, 0.0, 82.61879104, 34.0, 4.0],
        [88.89712609, 68.0, 0.0, 101.13223325, 38.0, 0.0, 0.0, 18.0, 4.0],
        [77.63250107, 11.0, 0.0, 103.59746386, 34.0, 0.0, 95.80275375, 31.0, 0.0],
        [97.42186916, 13.0, 0.0, 0.0, 13.0, 3.0, 91.53794969, 0.0, 3.0],
        [0.0, 12.0, 10.0, 99.11445303, 28.0, 0.0, 95.20557595, 8.0, 0.0],
    ]
)


def test_rdst_on_basic_motions():
    """Test of ShapeletTransform on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomDilatedShapeletTransform(max_shapelets=3, random_state=0)
    st.fit(X_train[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train[indices])
    assert_array_almost_equal(data, shapelet_transform_basic_motions_data, decimal=4)
