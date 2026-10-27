"""Tests for the rescaling transformers."""

import numpy as np
import pytest

from aeon.testing.data_generation._collection import make_example_3d_numpy_list
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


def _np_array(shape, constant=False):
    """Create a dummy numpy array with the given shape."""
    if constant:
        return np.ones(shape)
    return np.arange(np.prod(shape)).reshape(shape)


GLOBAL_SCALER_TESTS = [
    [_np_array((2, 5)), (0, 1), (1, 1)],
    [_np_array((2, 5), constant=True), (0, 1), (1, 1)],
    [_np_array((2, 3, 4)), (0, -1), (1, 3, 1)],
    [_np_array((2, 3, 4), constant=True), (0, -1), (1, 3, 1)],
    [
        make_example_3d_numpy_list(
            2,
            3,
            return_y=False,
        ),
        (1,),
        (3, 1),
    ],
]


@pytest.mark.parametrize("X, reduced_axis, target_shape", GLOBAL_SCALER_TESTS)
def test_global_z_norm(X, reduced_axis, target_shape):
    """Test GlobalNormalizer on regular array layouts."""
    normaliser = GlobalNormalizer(2, 3)
    X_transformed = normaliser.fit_transform(X)

    assert normaliser.x_means.shape == target_shape
    assert normaliser.x_stds.shape == target_shape

    # check that the transformed data has the desired characteristics
    X_transformed_ = X_transformed
    X_ = X
    if isinstance(X_transformed, list):
        X_transformed_ = np.concatenate(X_transformed_, axis=1)
        X_ = np.concatenate(X_, axis=1)

    mean = np.mean(X_transformed_, axis=reduced_axis)
    std = np.std(X_transformed_, axis=reduced_axis)
    assert np.allclose(mean, 2)
    if np.allclose(X_, 1):
        assert np.allclose(std, 0)
    else:
        assert np.allclose(std, 3)

    # check that the inverse transform recovers the original data
    X_inv = normaliser.inverse_transform(X_transformed)
    if isinstance(X_inv, list):
        X_inv = np.concatenate(X_inv, axis=1)
    assert np.allclose(X_, X_inv)


@pytest.mark.parametrize("X, reduced_axis, target_shape", GLOBAL_SCALER_TESTS)
def test_global_centering(X, reduced_axis, target_shape):
    """Test GlobalCenterer on regular array layouts."""
    centerer = GlobalCenterer(2)
    X_transformed = centerer.fit_transform(X)

    assert centerer.x_means.shape == target_shape

    X_transformed_ = X_transformed
    X_ = X
    if isinstance(X_transformed, list):
        X_transformed_ = np.concatenate(X_transformed_, axis=1)
        X_ = np.concatenate(X_, axis=1)

    # check that the transformed data has the desired characteristics
    mean = np.mean(X_transformed_, axis=reduced_axis)
    assert np.allclose(mean, 2)

    # check that the inverse transform recovers the original data
    X_inv = centerer.inverse_transform(X_transformed)
    if isinstance(X_inv, list):
        X_inv = np.concatenate(X_inv, axis=1)
    assert np.allclose(X_, X_inv)


@pytest.mark.parametrize("X, reduced_axis, target_shape", GLOBAL_SCALER_TESTS)
def test_global_min_max(X, reduced_axis, target_shape):
    """Test GlobalMinMaxScaler on regular array layouts."""
    minmax = GlobalMinMaxScaler(-1, 2)
    X_transformed = minmax.fit_transform(X)

    assert minmax.x_mins.shape == target_shape
    assert minmax.x_maxs.shape == target_shape

    X_transformed_ = X_transformed
    X_ = X
    if isinstance(X_transformed_, list):
        X_transformed_ = np.concatenate(X_transformed_, axis=1)
        X_ = np.concatenate(X_, axis=1)

    # check that the transformed data has the desired characteristics
    min_val = np.min(X_transformed_, axis=reduced_axis)
    max_val = np.max(X_transformed_, axis=reduced_axis)
    assert np.allclose(min_val, -1)
    if np.allclose(X_, 1):
        assert np.allclose(max_val, -1)
    else:
        assert np.allclose(max_val, 2)

    # check that the inverse transform recovers the original data
    X_inv = minmax.inverse_transform(X_transformed)
    if isinstance(X_inv, list):
        X_inv = np.concatenate(X_inv, axis=1)
    assert np.allclose(X_, X_inv)
