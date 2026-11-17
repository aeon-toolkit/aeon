"""Tests for the rescaling transformers."""

import numpy as np
import pytest

from aeon.testing.data_generation._collection import make_example_3d_numpy_list
from aeon.transformations.collection._rescale import Centerer, MinMaxScaler, Normalizer


def _np_array(shape, constant=False):
    """Create a dummy numpy array with the given shape."""
    if constant:
        return np.full(shape, np.random.rand())
    return np.random.normal(size=shape)


SCALER_TESTS = [
    ################################################
    # 3D arrays:
    (_np_array((2, 3, 4)), None, -1),  # (sample, channels, time)
    (_np_array((2, 3, 4)), 1, 1),  # (sample, time, channels)
    ################################################
    # Unequal length series:
    # - n_sample of 2D (channels, time)
    (make_example_3d_numpy_list(2, 3, return_y=False), 1, 1),
    # - n_sample of 2D (time, channels)
    (make_example_3d_numpy_list(2, 3, return_y=False), 0, 0),
    ################################################
    # 2D arrays:
    # (channels, time) = 1 multivariate or (samples, time) = n univariate
    (_np_array((2, 5)), -1, -1),
    (_np_array((2, 5)), 0, 0),  # (time, channels) = 1 multivariate
    ################################################
    # constant cases
    (_np_array((2, 3, 4), True), None, -1),  # (sample, channels, time)
    (_np_array((2, 5), True), 1, 1),
]


@pytest.mark.parametrize("X, axis, reduced_axis", SCALER_TESTS)
def test_z_norm(X, axis, reduced_axis):
    """Test Normalizer on regular array layouts."""
    if axis is None:
        normaliser = Normalizer()
    else:
        normaliser = Normalizer(axis=axis)

    out = normaliser.transform(X)

    if isinstance(X, list):
        assert all(isinstance(out[i], np.ndarray) for i in range(len(out)))
        assert all([out[i].shape == X[i].shape for i in range(len(out))])
        out = np.concatenate(out, axis=1)
    else:
        assert isinstance(out, np.ndarray)
        assert out.shape == X.shape

    mean = np.mean(out, axis=reduced_axis)
    std = np.std(out, axis=reduced_axis)

    assert np.allclose(mean, 0)
    is_X_constant = np.all([np.allclose(x, x.flat[0]) for x in X])
    if not is_X_constant:
        assert np.allclose(std, 1)
    else:
        assert np.allclose(std, 0)


@pytest.mark.parametrize("X, axis, reduced_axis", SCALER_TESTS)
def test_centering(X, axis, reduced_axis):
    """Test Centerer on regular array layouts."""
    if axis is None:
        centerer = Centerer()
    else:
        centerer = Centerer(axis=axis)

    out = centerer.transform(X)

    if isinstance(X, list):
        assert all(isinstance(out[i], np.ndarray) for i in range(len(out)))
        assert all([out[i].shape == X[i].shape for i in range(len(out))])
        out = np.concatenate(out, axis=1)
    else:
        assert isinstance(out, np.ndarray)
        assert out.shape == X.shape

    mean = np.mean(out, axis=reduced_axis)

    assert np.allclose(mean, 0)


@pytest.mark.parametrize("X, axis, reduced_axis", SCALER_TESTS)
def test_min_max(X, axis, reduced_axis):
    """Test MinMaxScaler on regular array layouts."""
    if axis is None:
        minmax = MinMaxScaler()
    else:
        minmax = MinMaxScaler(axis=axis)

    out = minmax.transform(X)

    if isinstance(X, list):
        assert all(isinstance(out[i], np.ndarray) for i in range(len(out)))
        assert all([out[i].shape == X[i].shape for i in range(len(out))])
        out = np.concatenate(out, axis=1)
    else:
        assert isinstance(out, np.ndarray)
        assert out.shape == X.shape

    min_val = np.min(out, axis=reduced_axis)
    max_val = np.max(out, axis=reduced_axis)

    assert np.allclose(min_val, 0)
    is_X_constant = np.all([np.allclose(x, x.flat[0]) for x in X])
    if not is_X_constant:
        assert np.allclose(max_val, 1)
    else:
        assert np.allclose(max_val, 0)


def test_min_max_invalid_range():
    """Test that MinMaxScaler raises ValueError when min >= max."""
    X = _np_array((2, 3, 4))
    with pytest.raises(ValueError, match="should be less than max value"):
        minmax = MinMaxScaler(min=1, max=0)
        minmax.transform(X)
