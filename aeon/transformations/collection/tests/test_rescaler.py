"""Tests for the rescaling transformers."""

import numpy as np
import pytest

from aeon.transformations.collection._rescale import (
    Centerer,
    GlobalCenterer,
    GlobalMinMaxScaler,
    GlobalNormalizer,
    MinMaxScaler,
    Normalizer,
)


def test_z_norm():
    """Test the Normalize class.

    This function creates a 3D numpy array, applies z-normalization using the
    Normalise class, and asserts that the transformed data has a mean close to 0 and a
    standard deviation close to 1 along the specified axis.
    """
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalizer()
    X_transformed = normaliser._transform(X)

    mean = np.mean(X_transformed, axis=-1)
    std = np.std(X_transformed, axis=-1)

    assert np.allclose(mean, 0)
    assert np.allclose(std, 1)


def test_centering():
    """Test the Centerer class."""
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    std = Centerer()
    X_transformed = std._transform(X)

    mean = np.mean(X_transformed, axis=-1)

    assert np.allclose(mean, 0)


def test_min_max():
    """Test the MinMaxScaler class."""
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    minmax = MinMaxScaler()
    X_transformed = minmax._transform(X)

    min_val = np.min(X_transformed, axis=-1)
    max_val = np.max(X_transformed, axis=-1)

    assert np.allclose(min_val, 0)
    assert np.allclose(max_val, 1)
    with pytest.raises(ValueError, match="should be less than max value"):
        minmax = MinMaxScaler(min=1, max=0)
        X_transformed = minmax._transform(X)


def _np_array(shape):
    """Create a dummy numpy array with the given shape."""
    return np.arange(np.prod(shape)).reshape(shape)


def _np_list(n, shape):
    """Create a dummy list of numpy arrays with the given shape."""
    p = np.prod(shape)
    return [np.arange(p).reshape(shape) + i * p for i in range(n)]


@pytest.mark.parametrize(
    "X, scaler_axis, reduce_axis",
    [
        ################################################
        # 3D arrays:
        (_np_array((2, 3, 4)), None, (0, -1)),  # (sample, channels, time)
        (_np_array((2, 3, 4)), 1, (0, 1)),  # (sample, time, channels)
        ################################################
        # Unequal length series:
        (_np_list(2, (3, 4)), None, (0, -1)),  # list of (channels, time)
        ################################################
        # 2D arrays:
        (_np_array((2, 5)), None, 1),  # (channels, time) = 1 multivariate
        (_np_array((2, 5)), 0, 0),  # (time, channels) = 1 multivariate
        (_np_array((2, 5)), (0, 1), (0, 1)),  # (samples, time)  = n univariate
        ################ EXOTIC CASES ! ################
        # 1D arrays: (time,) = 1 univariate
        (_np_array((10,)), None, -1),
        ################################################
        # 4D arrays: works thanks to generalisation but
        # few use cases for this in practice ...
        (_np_array((2, 3, 4, 5)), None, (0, 3)),  # (sample, channels, features, time)
        (_np_array((2, 3, 4, 5)), 2, (0, 2)),  # (sample, channels, time, features)
        (_np_array((2, 3, 4, 5)), 1, (0, 1)),  # (sample, time, channels, features)
        (_np_array((2, 3, 4, 5)), (2, 3), (0, 2, 3)),  # Weird example
    ],
)
def test_global_z_norm(X, scaler_axis, reduce_axis):
    """Test GlobalNormalizer on regular array layouts."""
    if scaler_axis is None:
        normaliser = GlobalNormalizer(2, 3)
    else:
        normaliser = GlobalNormalizer(2, 3, axis=scaler_axis)

    X_transformed = normaliser.fit_transform(X)

    mean = np.mean(X_transformed, axis=reduce_axis)
    std = np.std(X_transformed, axis=reduce_axis)

    assert np.allclose(mean, 2)
    assert np.allclose(std, 3)

    X_inv = normaliser.inverse_transform(X_transformed)
    assert np.allclose(X, X_inv)


@pytest.mark.parametrize(
    "X, scaler_axis, reduce_axis",
    [
        ################################################
        # 3D arrays:
        (_np_array((2, 3, 4)), None, (0, -1)),  # (sample, channels, time)
        (_np_array((2, 3, 4)), 1, (0, 1)),  # (sample, time, channels)
        ################################################
        # Unequal length series:
        (_np_list(2, (3, 4)), None, (0, -1)),  # list of (channels, time)
        ################################################
        # 2D arrays:
        (_np_array((2, 5)), None, 1),  # (channels, time) = 1 multivariate
        (_np_array((2, 5)), 0, 0),  # (time, channels) = 1 multivariate
        (_np_array((2, 5)), (0, 1), (0, 1)),  # (samples, time)  = n univariate
        ################ EXOTIC CASES ! ################
        # 1D arrays: (time,) = 1 univariate
        (_np_array((10,)), None, -1),
        ################################################
        # 4D arrays: works thanks to generalisation but
        # few use cases for this in practice ...
        (_np_array((2, 3, 4, 5)), None, (0, 3)),  # (sample, channels, features, time)
        (_np_array((2, 3, 4, 5)), 2, (0, 2)),  # (sample, channels, time, features)
        (_np_array((2, 3, 4, 5)), 1, (0, 1)),  # (sample, time, channels, features)
        (_np_array((2, 3, 4, 5)), (2, 3), (0, 2, 3)),  # Weird example
    ],
)
def test_global_minmax_norm(X, scaler_axis, reduce_axis):
    """Test GlobalMinMaxScaler on regular array layouts."""
    if scaler_axis is None:
        normaliser = GlobalMinMaxScaler(-1, 2)
    else:
        normaliser = GlobalMinMaxScaler(-1, 2, axis=scaler_axis)

    X_transformed = normaliser.fit_transform(X)

    mini = np.min(X_transformed, axis=reduce_axis)
    maxi = np.max(X_transformed, axis=reduce_axis)
    assert np.allclose(mini, -1)
    assert np.allclose(maxi, 2)

    X_inv = normaliser.inverse_transform(X_transformed)
    assert np.allclose(X, X_inv)


@pytest.mark.parametrize(
    "X, scaler_axis, reduce_axis",
    [
        ################################################
        # 3D arrays:
        (_np_array((2, 3, 4)), None, (0, -1)),  # (sample, channels, time)
        (_np_array((2, 3, 4)), 1, (0, 1)),  # (sample, time, channels)
        ################################################
        # Unequal length series:
        (_np_list(2, (3, 4)), None, (0, -1)),  # list of (channels, time)
        ################################################
        # 2D arrays:
        (_np_array((2, 5)), None, 1),  # (channels, time) = 1 multivariate
        (_np_array((2, 5)), 0, 0),  # (time, channels) = 1 multivariate
        (_np_array((2, 5)), (0, 1), (0, 1)),  # (samples, time)  = n univariate
        ################ EXOTIC CASES ! ################
        # 1D arrays: (time,) = 1 univariate
        (_np_array((10,)), None, -1),
        ################################################
        # 4D arrays: works thanks to generalisation but
        # few use cases for this in practice ...
        (_np_array((2, 3, 4, 5)), None, (0, 3)),  # (sample, channels, features, time)
        (_np_array((2, 3, 4, 5)), 2, (0, 2)),  # (sample, channels, time, features)
        (_np_array((2, 3, 4, 5)), 1, (0, 1)),  # (sample, time, channels, features)
        (_np_array((2, 3, 4, 5)), (2, 3), (0, 2, 3)),  # Weird example
    ],
)
def test_global_center_norm(X, scaler_axis, reduce_axis):
    """Test GlobalCenterer on regular array layouts."""
    if scaler_axis is None:
        normaliser = GlobalCenterer(2)
    else:
        normaliser = GlobalCenterer(2, axis=scaler_axis)

    X_transformed = normaliser.fit_transform(X)

    mean = np.mean(X_transformed, axis=reduce_axis)
    assert np.allclose(mean, 2)

    X_inv = normaliser.inverse_transform(X_transformed)
    assert np.allclose(X, X_inv)
