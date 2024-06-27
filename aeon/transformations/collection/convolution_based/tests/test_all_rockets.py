"""Rocket test code."""

import numpy as np
import pytest

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)

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
    "Rocket": [1.0, 1.3438051, 0.53333336],
    "MiniRocket": [0.4, 1.0, 1.0],
    "MultiRocket": [0.4, 1.0, 1.0],
}
expected_features = {
    "Rocket": 200,
    "MultiRocket": 672,
    "MiniRocket": 84,
    "MiniRocketMultivariate": 84,
}
expected_multi = {
    "Rocket": [0.6363636, 0.74931127, 0.0],
    "MiniRocketMultivariate": [0.27272728, 0.33333334, 0.8181818],
}


@pytest.mark.parametrize(
    "transform",
    ["Rocket", "MiniRocket"],
)
def test_rocket_on_univariate(transform):
    """Test of Rocket on gun point."""
    # Create random univariate training data
    X = uni_test_data
    if transform == "Rocket":
        ROCKET = Rocket(num_kernels=100, random_state=0)
    elif transform == "MiniRocket":
        ROCKET = MiniRocket(num_kernels=100, random_state=0)
    # elif transform == "MultiRocket":
    #     ROCKET = MultiRocket(num_kernels=100, random_state=0)
    ROCKET.fit(X)
    # transform training data
    X_trans = ROCKET.transform(X)
    # test shape of transformed training data -> (number of training
    # examples, num_kernels * 2)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features[transform]))
    np.testing.assert_almost_equal(
        np.array(expected_uni[transform]),
        np.array([X_trans[0][0], X_trans[1][5], X_trans[3][80]]),
    )
    # Test fit_transform the same
    X_trans2 = ROCKET.fit_transform(X)
    assert X_trans[2][3] == X_trans2[2][3]
    assert X_trans[0][80] == X_trans2[0][80]
    assert X_trans[3][55] == X_trans2[3][55]


@pytest.mark.parametrize("transform", ["Rocket", "MiniRocketMultivariate"])
def test_rocket_on_multivariate(transform):
    """Test of Rocket on gun point."""
    # Create random univariate training data
    X = multi_test_data
    if transform == "Rocket":
        ROCKET = Rocket(num_kernels=100, random_state=0)
    elif transform == "MiniRocketMultivariate":
        ROCKET = MiniRocketMultivariate(num_kernels=100, random_state=0)
    ROCKET.fit(X)
    # transform training data
    X_trans = ROCKET.transform(X)
    # test shape of transformed training data -> (number of training
    # examples, num_kernels * 2)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features[transform]))
    np.testing.assert_almost_equal(
        np.array(expected_multi[transform]),
        np.array([X_trans[0][0], X_trans[1][5], X_trans[3][80]]),
    )
    # Test fit_transform the same
    X_trans2 = ROCKET.fit_transform(X)
    assert X_trans[2][3] == X_trans2[2][3]
    assert X_trans[0][80] == X_trans2[0][80]
    assert X_trans[3][55] == X_trans2[3][55]


def test_normalise_rocket():
    """Test normalization with Rocket."""
    arr = np.random.random(size=(10, 1, 100))
    rocket = Rocket(num_kernels=200, normalise=True)
    trans = rocket.fit_transform(arr)
    assert trans.shape == (10, 400)
    rocket = MultiRocket(num_kernels=200, normalise=True)
    trans = rocket.fit_transform(arr)
    assert trans.shape == (10, 1344)


expected_unit_test = {}
expected_unit_test["MiniRocket"] = np.array(
    [
        [0.5833333, 0.7083333, 0.16666667, 0.5833333, 0.8333333],
        [0.5416667, 0.7083333, 0.16666667, 0.5416667, 0.7916667],
        [0.5833333, 0.7083333, 0.20833333, 0.5833333, 0.7916667],
        [0.45833334, 0.6666667, 0.16666667, 0.5, 0.75],
        [0.5416667, 0.7916667, 0.125, 0.5416667, 0.8333333],
    ]
)

expected_basic_motions = {}
expected_basic_motions["MiniRocket"] = np.array(
    [
        [0.38, 0.76, 0.15, 0.53, 0.91],
        [0.43, 0.7, 0.09, 0.53, 0.95],
        [0.46, 0.63, 0.25, 0.54, 0.87],
        [0.3, 0.79, 0.04, 0.47, 1],
        [0.45, 0.78, 0.02, 0.55, 1],
    ]
)


def test_datatype_input():
    """Test rocket variants accept all input types."""
    uni_rockets = [
        Rocket(num_kernels=100),
        MiniRocket(num_kernels=100),
        MultiRocket(num_kernels=100),
    ]

    shape = (10, 1, 20)
    types = [np.float32, np.float64, np.int32, np.int64]
    for r in uni_rockets:
        for t in types:
            X = np.random.rand(*shape).astype(t)
            r.fit_transform(X)
            r.fit(X)
            r.transform(X)
    multi_rockets = [
        MiniRocketMultivariate(num_kernels=100),
        MultiRocketMultivariate(num_kernels=100),
    ]
    shape = (10, 3, 20)
    for r in multi_rockets:
        for t in types:
            X = np.random.rand(*shape).astype(t)
            r.fit_transform(X)
            r.fit(X)
            r.transform(X)


def test_expected_unit_test():
    """Test MiniRocket on unit test data."""
    X, _ = load_unit_test(split="train")
    mr = MiniRocket(random_state=0)
    X2 = mr.fit_transform(X)
    np.testing.assert_allclose(X2[:5, :5], expected_unit_test["MiniRocket"], rtol=1e-6)


def test_expected_basic_motions():
    """Test MiniRocket on unit test data."""
    X, _ = load_basic_motions(split="train")
    mr = MiniRocket(random_state=0)
    X2 = mr.fit_transform(X)

    np.testing.assert_allclose(
        X2[:5, :5], expected_basic_motions["MiniRocket"], rtol=1e-6
    )
