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
    [_np_array((2, 5)), (0, 1)],
    [_np_array((2, 5), constant=True), (0, 1)],
    [_np_array((2, 3, 4)), (0, -1)],
    [_np_array((2, 3, 4), constant=True), (0, -1)],
    [make_example_3d_numpy_list(2, 3, return_y=False), 1],
]


@pytest.mark.parametrize("X, reduced_axis", GLOBAL_SCALER_TESTS)
def test_global_z_norm(X, reduced_axis):
    """Test GlobalNormalizer on regular array layouts."""
    normaliser = GlobalNormalizer(2, 3)
    X_transformed = normaliser.fit_transform(X)

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


@pytest.mark.parametrize("X, reduced_axis", GLOBAL_SCALER_TESTS)
def test_global_centering(X, reduced_axis):
    """Test GlobalCenterer on regular array layouts."""
    centerer = GlobalCenterer(2)
    X_transformed = centerer.fit_transform(X)

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


@pytest.mark.parametrize("X, reduced_axis", GLOBAL_SCALER_TESTS)
def test_global_min_max(X, reduced_axis):
    """Test GlobalMinMaxScaler on regular array layouts."""
    minmax = GlobalMinMaxScaler(-1, 2)
    X_transformed = minmax.fit_transform(X)

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


@pytest.mark.parametrize(
    "Scaler", [GlobalCenterer, GlobalMinMaxScaler, GlobalNormalizer]
)
def test_global_scaler_variable_input_layout(Scaler):
    """Test GlobalCenterer on variable-dim input."""
    X3 = np.random.normal(size=(6, 2, 20))
    Xl = [np.random.normal(size=(2, n)) for n in (15, 20, 25)]

    scaler_X3 = Scaler().fit(X3)
    out = scaler_X3.transform(Xl)
    assert isinstance(out, list)
    assert all(isinstance(x, np.ndarray) for x in out)
    assert all(out[i].shape == Xl[i].shape for i in range(len(out)))

    out = scaler_X3.inverse_transform(Xl)
    assert isinstance(out, list)
    assert all(isinstance(x, np.ndarray) for x in out)
    assert all(out[i].shape == Xl[i].shape for i in range(len(out)))

    scaler_Xl = Scaler().fit(Xl)
    out = scaler_Xl.transform(X3)
    assert isinstance(out, np.ndarray)
    assert out.shape == X3.shape

    out = scaler_Xl.inverse_transform(X3)
    assert isinstance(out, np.ndarray)
    assert out.shape == X3.shape

    X2 = np.random.normal(size=(6, 20))
    X3 = np.random.normal(size=(6, 1, 20))

    scaler_X2 = Scaler().fit(X2)
    out = scaler_X2.transform(X3)
    assert isinstance(out, np.ndarray)
    assert out.shape == X3.shape

    scaler_X3 = Scaler().fit(X3)
    out = scaler_X3.transform(X2)
    assert isinstance(out, np.ndarray)
    assert out.shape == X2.shape
