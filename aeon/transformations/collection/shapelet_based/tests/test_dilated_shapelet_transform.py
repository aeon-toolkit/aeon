"""Tests for dilated shapelet transform functions."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
)

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.distances import manhattan_distance
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy_list,
)
from aeon.transformations.collection.shapelet_based._dilated_shapelet_transform import (
    RandomDilatedShapeletTransform,
    compute_shapelet_dist_vector,
    compute_shapelet_features,
)
from aeon.utils.numba.general import get_dilated_subsequences, is_prime

DATATYPES = ["int64", "float64"]


def _reference_data(case):
    """Return the small datasets used by the reference tests."""
    if case == "unit_test":
        X, y = load_unit_test(split="train")
        indices = np.random.RandomState(0).choice(len(y), 5, replace=False)
        return X[indices], y[indices], {}
    if case == "basic_motions":
        X, y = load_basic_motions(split="train")
        indices = np.random.RandomState(4).choice(len(y), 5, replace=False)
        return X[indices], y[indices], {}
    # Multivariate series of unequal length
    X, y = make_example_3d_numpy_list(
        n_cases=5,
        n_channels=2,
        min_n_timepoints=30,
        max_n_timepoints=40,
        random_state=0,
    )
    return X, y, {"shapelet_lengths": [5, 7]}


# Shapelets and transformed data obtained with
# RandomDilatedShapeletTransform(max_shapelets=4, random_state=0) on the datasets of
# _reference_data. They pin both the sampling and the feature computation, so that a
# change of implementation which alters either of them is caught.
REFERENCE_SHAPELETS = {
    "unit_test": {
        "startpoints": [0, 6, 8, 13],
        "lengths": [11, 11, 11, 11],
        "dilations": [1, 1, 1, 1],
        "normalises": [False, False, True, True],
        "classes": [1, 0, 0, 1],
        "thresholds": [814.41930467, 2109.03667492, 2.56612734, 3.91406779],
    },
    "basic_motions": {
        "startpoints": [18, 1, 64, 2],
        "lengths": [11, 11, 11, 11],
        "dilations": [4, 1, 1, 1],
        "normalises": [False, False, True, True],
        "classes": [1, 3, 3, 0],
        "thresholds": [217.07350248, 54.09640355, 39.02875728, 57.59835024],
    },
    "unequal": {
        "startpoints": [24, 17, 0, 11],
        "lengths": [7, 7, 7, 5],
        "dilations": [1, 1, 3, 5],
        "normalises": [True, True, True, False],
        "classes": [0, 1, 0, 1],
        "thresholds": [11.29629834, 11.54865819, 11.16381163, 9.69510188],
    },
}

# Each row holds the (min, argmin, shapelet occurrence) features of the 4 shapelets
REFERENCE_TRANSFORM = {
    "unit_test": np.array(
        [
            [175, 0, 2, 1872, 6, 1, 1.2216, 8, 1, 2.0003, 13, 2],
            [946, 1, 0, 1582, 6, 1, 1.6112, 8, 2, 2.5269, 13, 1],
            [302, 0, 1, 2780, 6, 0, 2.7369, 7, 0, 0, 13, 2],
            [1467, 0, 0, 0, 6, 3, 0, 8, 1, 10.3812, 13, 0],
            [0, 0, 2, 855, 6, 2, 1.6068, 8, 1, 3.1499, 13, 1],
        ]
    ),
    "basic_motions": np.array(
        [
            [378.4535, 49, 0, 47.2864, 80, 6, 28.2159, 34, 8, 55.3703, 32, 1],
            [375.3853, 15, 0, 103.1230, 47, 0, 47.9410, 67, 0, 0, 2, 7],
            [389.4505, 30, 0, 38.3285, 25, 88, 36.3951, 20, 1, 53.6332, 72, 3],
            [0, 18, 4, 289.7186, 0, 0, 48.2920, 35, 0, 54.8160, 74, 4],
            [373.9575, 44, 0, 0, 1, 6, 0, 64, 13, 56.4145, 4, 1],
        ]
    ),
    "unequal": np.array(
        [
            [10.0646, 15, 2, 10.0037, 5, 2, 10.6568, 13, 2, 15.8547, 3, 0],
            [8.1854, 23, 5, 7.3494, 8, 2, 10.1489, 3, 3, 6.4994, 6, 3],
            [11.3015, 0, 0, 10.8491, 21, 2, 11.2485, 1, 0, 0, 11, 1],
            [0, 24, 3, 9.2461, 5, 1, 0, 0, 1, 13.7069, 3, 0],
            [9.8033, 4, 2, 0, 17, 3, 9.3214, 4, 1, 8.0997, 7, 1],
        ]
    ),
}


@pytest.mark.parametrize("case", list(REFERENCE_SHAPELETS))
def test_rdst_fit_matches_reference(case):
    """Extracted shapelets are unchanged for a fixed seed."""
    X, y, params = _reference_data(case)
    rdst = RandomDilatedShapeletTransform(max_shapelets=4, random_state=0, **params)
    rdst.fit(X, y)
    _, startpoints, lengths, dilations, thresholds, normalises, _, _, classes = (
        rdst.shapelets_
    )
    expected = REFERENCE_SHAPELETS[case]
    assert startpoints.tolist() == expected["startpoints"]
    assert lengths.tolist() == expected["lengths"]
    assert dilations.tolist() == expected["dilations"]
    assert normalises.tolist() == expected["normalises"]
    assert classes.tolist() == expected["classes"]
    assert_allclose(thresholds, expected["thresholds"], rtol=1e-6)


@pytest.mark.parametrize("case", list(REFERENCE_TRANSFORM))
def test_rdst_transform_matches_reference(case):
    """Transformed data is unchanged for a fixed seed."""
    X, y, params = _reference_data(case)
    rdst = RandomDilatedShapeletTransform(max_shapelets=4, random_state=0, **params)
    X_t = rdst.fit(X, y).transform(X)
    assert_array_almost_equal(X_t, REFERENCE_TRANSFORM[case], decimal=3)


def test_rdst_transform_float32_input():
    """float32 input gives the same features as float64 input."""
    X, y, _ = _reference_data("basic_motions")
    rdst = RandomDilatedShapeletTransform(max_shapelets=4, random_state=0).fit(X, y)
    X_t = rdst.transform(X)
    X_t32 = rdst.transform(X.astype(np.float32))
    # atol for the near zero minimum distances, which float32 rounds to ~1e-6
    assert_allclose(X_t32, X_t, rtol=1e-4, atol=1e-4)


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
    X_subs = get_dilated_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, threshold)

    # On some occasion, float32 precision with fasmath returns things like
    # 2.1835059227370834e-07 instead of 0
    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 0.0
    assert SO == 3.0

    dilation = 2
    threshold = 0.1
    X_subs = get_dilated_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, threshold)

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 7.0
    assert SO == 1.0

    dilation = 4
    threshold = 2
    X_subs = get_dilated_subsequences(X, length, dilation)
    _min, _argmin, SO = compute_shapelet_features(X_subs, values, threshold)

    assert_almost_equal(_min, 0.0, decimal=4)
    assert _argmin == 3.0
    assert SO == 3.0


@pytest.mark.parametrize("dtype", DATATYPES)
def test_compute_shapelet_dist_vector(dtype):
    """Test computation of shapelet distance vector."""
    X = make_example_2d_numpy_series(n_timepoints=50, n_channels=3, random_state=0)
    X = (X * 100).astype(dtype)
    for length in [3, 5]:
        for dilation in [1, 3, 5]:
            values = make_example_2d_numpy_series(
                n_timepoints=length, n_channels=3, random_state=1
            )
            values = (values * 100).astype(dtype)
            X_subs = get_dilated_subsequences(X, length, dilation)
            d_vect = compute_shapelet_dist_vector(X_subs, values)
            true_vect = np.zeros(X.shape[1] - (length - 1) * dilation)
            for i_sub in range(true_vect.shape[0]):
                _idx = [i_sub + j * dilation for j in range(length)]
                _sub = X[:, _idx]
                true_vect[i_sub] += manhattan_distance(values, _sub)
            assert_array_almost_equal(d_vect, true_vect)
