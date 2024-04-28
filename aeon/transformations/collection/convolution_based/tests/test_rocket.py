"""Rocket test code."""

import numpy as np
import pytest

from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
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
    elif transform == "MultiRocket":
        ROCKET = MultiRocket(num_kernels=100, random_state=0)
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
