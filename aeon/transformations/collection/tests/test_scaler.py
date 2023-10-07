"""Test the StandardScaler for time series."""

import numpy as np
import pytest

from aeon.transformations.collection import TimeSeriesScaler

X = np.array([[[0, 0, 0, 0], [10, 10, 10, 10]], [[1, 2, 3, 4], [5, 2, 3, 7]]])
expected = np.array(
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [-1.34164079, -0.4472136, 0.4472136, 1.34164079],
            [0.39056673, -1.1717002, -0.65094455, 1.43207802],
        ],
    ]
)

expected_no_std = np.array(
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [[-1.5, -0.5, 0.5, 1.5], [0.75, -2.25, -1.25, 2.75]],
    ]
)

expected_no_mean = np.array(
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        [
            [0.89442719, 1.78885438, 2.68328157, 3.57770876],
            [2.60377822, 1.04151129, 1.56226693, 3.64528951],
        ],
    ]
)

expected_no_std = np.array(
    [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [[-1.5, -0.5, 0.5, 1.5], [0.75, -2.25, -1.25, 2.75]],
    ]
)

x1 = np.array([[0, 0, 0], [10, 10, 10]])
x2 = np.array([[1, 2, 3, 4], [5, 2, 3, 7]])
X_unequal = [x1, x2]
a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


@pytest.mark.parametrize("with_mean", [True, False])
def test_standard_scaler_mean(with_mean):
    """Test Standard scaler with and without mean, equal and unequal length."""
    # Equal length 2 channels, 2 cases
    sc = TimeSeriesScaler(with_mean=with_mean)
    X2 = sc.fit_transform(X)
    assert isinstance(X2, np.ndarray)
    assert X2.shape == X.shape
    if with_mean:
        np.testing.assert_almost_equal(expected, X2)
    else:
        np.testing.assert_almost_equal(expected_no_mean, X2)
    if with_mean:
        X2 = sc.fit_transform(X_unequal)
        assert isinstance(X2, list)
        np.testing.assert_almost_equal(X2[0], a)


@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler_std_dev(with_std):
    """Test Standard scaler with and without std."""
    # Equal length 2 channels, 2 cases
    sc = TimeSeriesScaler(with_std=with_std)
    X2 = sc.fit_transform(X)
    assert X2.shape == X.shape
    if with_std:
        np.testing.assert_almost_equal(expected, X2)
    else:
        np.testing.assert_almost_equal(expected_no_std, X2)
