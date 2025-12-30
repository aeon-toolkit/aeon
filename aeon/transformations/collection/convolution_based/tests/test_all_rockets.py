"""Rocket test code."""

import numpy as np
import pytest

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)
from aeon.transformations.collection.convolution_based._minirocket import _PPV

# Data used to test correctness of transform
uni_test_data = np.array(
    [
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
        [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4]],
        [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]],
        [[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]],
    ]
)
multi_test_data = np.array(
    [
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7]],
        [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]],
        [[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], [4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 8]],
    ]
)

expected_uni = {
    "Rocket": [0.13333334, 1.622878, 0.8666667],
    "MiniRocket": [0.4, 1.0, 0.93333334],
    "MultiRocket": [0.4, 1.0, 0.93333334],
}
expected_features = {
    "Rocket": 200,
    "MiniRocket": 84,
    "MultiRocket": 672,
}
expected_multi = {
    "Rocket": [0.8181818, 5.735185, 0.6],
    "MiniRocket": [0.36363637, 0.0, 0.90909094],
    "MultiRocket": [0.36363637, 1.0, 0.90909094],
}


@pytest.mark.parametrize(
    "transform",
    ["Rocket", "MiniRocket", "MultiRocket"],
)
def test_rocket_on_univariate(transform):
    """Test of Rocket on gun point."""
    # Create random univariate training data
    X = uni_test_data
    if transform == "Rocket":
        rocket = Rocket(n_kernels=100, random_state=0)
    elif transform == "MiniRocket":
        rocket = MiniRocket(n_kernels=100, random_state=0)
    elif transform == "MultiRocket":
        rocket = MultiRocket(n_kernels=100, random_state=0)
    rocket.fit(X)
    # transform training data
    X_trans = rocket.transform(X)
    # test shape of transformed training data -> (number of training
    # examples, n_kernels * 2)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features[transform]))
    np.testing.assert_almost_equal(
        np.array(expected_uni[transform]),
        np.array([X_trans[0][0], X_trans[1][5], X_trans[3][80]]),
    )
    # Test fit_transform the same
    X_trans2 = rocket.fit_transform(X)
    assert X_trans[2][3] == X_trans2[2][3]
    assert X_trans[0][80] == X_trans2[0][80]
    assert X_trans[3][55] == X_trans2[3][55]


#
@pytest.mark.parametrize("transform", ["Rocket", "MiniRocket", "MultiRocket"])  #
def test_rocket_on_multivariate(transform):
    """Test of Rocket on gun point."""
    # Create random univariate training data
    X = multi_test_data
    if transform == "Rocket":
        rocket = Rocket(n_kernels=100, random_state=0)
    elif transform == "MiniRocket":
        rocket = MiniRocket(n_kernels=100, random_state=0)
    else:
        rocket = MultiRocket(n_kernels=100, random_state=0)

    rocket.fit(X)
    # transform training data
    X_trans = rocket.transform(X)
    # test shape of transformed training data -> (number of training
    # examples, n_kernels * 2)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features[transform]))
    np.testing.assert_almost_equal(
        np.array(expected_multi[transform]),
        np.array([X_trans[0][0], X_trans[1][5], X_trans[3][80]]),
    )
    # Test fit_transform the same
    X_trans2 = rocket.fit_transform(X)
    assert X_trans[2][3] == X_trans2[2][3]
    assert X_trans[0][80] == X_trans2[0][80]
    assert X_trans[3][55] == X_trans2[3][55]


def test_normalise_rocket():
    """Test normalization with Rocket."""
    arr = np.random.random(size=(10, 1, 100))
    rocket = Rocket(n_kernels=200, normalise=True)
    trans = rocket.fit_transform(arr)
    assert trans.shape == (10, 400)
    rocket = MultiRocket(n_kernels=200, normalise=True)
    trans = rocket.fit_transform(arr)
    assert trans.shape == (10, 1344)


expected_unit_test = {}
expected_unit_test["MiniRocket"] = np.array(
    [
        [0.5833333, 0.7083333, 0.1666667, 0.5833333, 0.8333333],
        [0.5416667, 0.7083333, 0.1666667, 0.5416667, 0.7916667],
        [0.5833333, 0.7083333, 0.2083333, 0.5833333, 0.7916667],
        [0.4583333, 0.6666667, 0.1666667, 0.5, 0.75],
        [0.5416667, 0.7916667, 0.125, 0.5416667, 0.8333333],
    ],
    dtype=np.float32,
)
expected_unit_test["Rocket"] = np.array(
    [
        [0.5, 2.507311, 0.416667, 7.098707, 0.875],
        [0.5, 2.00701, 0.416667, 6.768808, 0.958333],
        [0.5, 2.53585, 0.416667, 6.673825, 0.916667],
        [0.3333333, 1.328932, 0.4583333, 6.315244, 0.8333333],
        [0.5, 1.519787, 0.375, 6.620714, 0.875],
    ],
    dtype=np.float32,
)
expected_unit_test["MultiRocket"] = np.array(
    [
        [0.5833333, 0.7083333, 0.1666667, 0.5833333, 0.8333333],
        [0.5416667, 0.7083333, 0.1666667, 0.5416667, 0.7916667],
        [0.5833333, 0.7083333, 0.2083333, 0.5833333, 0.7916667],
        [0.4583333, 0.6666667, 0.1666667, 0.5, 0.75],
        [0.5416667, 0.7916667, 0.125, 0.5416667, 0.8333333],
    ],
    dtype=np.float32,
)

expected_basic_motions = {}
expected_basic_motions["MiniRocket"] = np.array(
    [
        [0.02, 1, 0, 0.59, 1],
        [0.04, 0.98, 0, 0.55, 0.99],
        [0, 1, 0, 0.47, 1],
        [0, 1, 0, 0.58, 1],
        [0, 1, 0, 0.57, 1],
    ],
    dtype=np.float32,
)
rockets = [
    Rocket(n_kernels=100),
    MultiRocket(n_kernels=100),
    MiniRocket(n_kernels=100),
]
types = [np.float32, np.float64, np.int32, np.int64]
data = [
    np.random.random((10, 20)),
    np.random.random((10, 1, 20)),
    np.random.random((10, 3, 20)),
]


@pytest.mark.parametrize("rocket", rockets)
@pytest.mark.parametrize("t", types)
@pytest.mark.parametrize("X", data)
def test_rocket_inputs(rocket, t, X):
    """Test rocket variants accept all input types."""
    X = X.astype(t)
    rocket.fit_transform(X)
    rocket.fit(X)
    X = rocket.transform(X)
    assert X.shape[0] == 10


@pytest.mark.parametrize("rocket", rockets)
def test_rocket_single_transform(rocket):
    """Test rockets can transform a single time series."""
    X = np.random.random((10, 1, 20))
    X2 = np.random.random((1, 20))
    rocket.fit(X)
    rocket.transform(X2)


def test_expected_unit_test():
    """Test MiniRocket on unit test data."""
    X, _ = load_unit_test(split="train")
    r = Rocket(random_state=0)
    mr = MiniRocket(random_state=0)
    mur = MultiRocket(random_state=0)
    X2 = r.fit_transform(X)
    X3 = mr.fit_transform(X)
    X4 = mur.fit_transform(X)
    np.testing.assert_allclose(X2[:5, :5], expected_unit_test["Rocket"], rtol=1e-4)
    np.testing.assert_allclose(X3[:5, :5], expected_unit_test["MiniRocket"], rtol=1e-4)
    np.testing.assert_allclose(
        X4[:5, :5], expected_unit_test["MultiRocket"], rtol=1e-4
    )


def test_expected_basic_motions():
    """Test MiniRocket on unit test data."""
    X, _ = load_basic_motions(split="train")
    mr = MiniRocket(random_state=0)
    X2 = mr.fit_transform(X)

    np.testing.assert_allclose(
        X2[:5, :5], expected_basic_motions["MiniRocket"], rtol=1e-4
    )


def test_ppv():
    """Test uncovered PPV function."""
    a = np.float32(10.0)
    b = np.float32(-5.0)
    assert _PPV(a, b) == 1
    assert _PPV(b, a) == 0
