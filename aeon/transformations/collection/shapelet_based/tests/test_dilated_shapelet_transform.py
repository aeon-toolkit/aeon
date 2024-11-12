"""Tests for dilated shapelet transform functions."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

# from aeon.datasets import load_basic_motions, load_unit_test
from aeon.datasets import load_basic_motions
from aeon.distances import manhattan_distance
from aeon.transformations.collection.shapelet_based._dilated_shapelet_transform import (
    RandomDilatedShapeletTransform,
    compute_shapelet_dist_vector,
    compute_shapelet_features,
)
from aeon.utils.numba.general import get_all_subsequences
from aeon.utils.numba.stats import is_prime

DATATYPES = ["int64", "float64"]


# The following test fail on MacOS due to an issue with the random seed.
#
# shapelet_transform_unit_test_data = np.array(
#     [
#         [1.90317756, 8.0, 2.0, 2.87919021, 10.0, 3.0, 0.0, 1.0, 1.0],
#         [2.16550181, 8.0, 2.0, 0.0, 10.0, 2.0, 1.52148128, 3.0, 1.0],
#         [0.0, 8.0, 1.0, 3.41218663, 10.0, 2.0, 1.00243477, 1.0, 2.0],
#         [2.76771406, 8.0, 2.0, 5.75682976, 10.0, 1.0, 1.66589725, 3.0, 1.0],
#         [2.95206323, 8.0, 2.0, 2.82417348, 10.0, 3.0, 0.91588726, 1.0, 1.0],
#     ]
# )
#
#
# def test_rdst_on_unit_test():
#     Test of ShapeletTransform on unit test data.
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train")
#     indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)
#
#     # fit the shapelet transform
#     st = RandomDilatedShapeletTransform(max_shapelets=3, random_state=0)
#     st.fit(X_train[indices], y_train[indices])
#
#     # assert transformed data is the same
#     data = st.transform(X_train[indices])
#     assert_array_almost_equal(data, shapelet_transform_unit_test_data, decimal=4)
#
#
# shapelet_transform_basic_motions_data = np.array(
#     [
#         [32.45712774, 25.0, 5.0, 58.52357949, 5.0, 0.0, 56.32267413, 21.0, 4.0],
#         [59.8154656, 69.0, 0.0, 64.16747582, 37.0, 0.0, 0.0, 18.0, 5.0],
#         [58.27369761, 11.0, 0.0, 67.49320392, 53.0, 0.0, 61.18423956, 31.0, 1.0],
#         [62.49300933, 13.0, 0.0, 0.0, 13.0, 5.0, 59.51080993, 34.0, 3.0],
#         [0.0, 12.0, 12.0, 64.73843849, 13.0, 0.0, 62.52577812, 8.0, 0.0],
#     ]
# )
#
#
# def test_rdst_on_basic_motions():
#     Test of ShapeletTransform on basic motions data.
#     # load basic motions data
#     X_train, y_train = load_basic_motions(split="train")
#     indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)
#
#     # fit the shapelet transform
#     st = RandomDilatedShapeletTransform(max_shapelets=3, random_state=0)
#     st.fit(X_train[indices], y_train[indices])
#
#     # assert transformed data is the same
#     data = st.transform(X_train[indices])
#     assert_array_almost_equal(data, shapelet_transform_basic_motions_data, decimal=4)
#


def test_shapelet_prime_dilation():
    """Test if dilations are prime numbers."""
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 3, replace=False)
    rdst = RandomDilatedShapeletTransform(
        max_shapelets=10, use_prime_dilations=True
    ).fit(X_train[indices], y_train[indices])
    dilations = rdst.shapelets_[2]
    assert np.all([d == 1 or is_prime(d) for d in dilations])


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_features(dtype):
    """Test computation of shapelet features."""
    X = np.asarray([[1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]], dtype=dtype)
    values = np.asarray([[1, 1, 2]], dtype=dtype)
    length = 3
    dilation = 1
    threshold = 0.01
    X_subs = get_all_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, threshold)

    # On some occasion, float32 precision with fasmath retruns things like
    # 2.1835059227370834e-07 instead of 0
    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 0.0
    assert SO == 3.0

    dilation = 2
    threshold = 0.1
    X_subs = get_all_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, threshold)

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 7.0
    assert SO == 1.0

    dilation = 4
    threshold = 2
    X_subs = get_all_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, threshold)

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 3.0
    assert SO == 3.0


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_dist_vector(dtype):
    """Test computation of shapelet distance vector."""
    X = np.random.rand(3, 50).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = np.random.rand(3, length).astype(dtype)
            X_subs = get_all_subsequences(X, length, dilation)
            d_vect = compute_shapelet_dist_vector(X_subs, values)
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                _sub = X[:, _idx]
                true_vect[i_sub] += manhattan_distance(values, _sub)
            assert_array_almost_equal(d_vect, true_vect)
