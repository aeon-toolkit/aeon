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


def test_global_z_norm():
    """Test the GlobalNormalizer class."""
    X = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9, 10], [10, 11, 12, 13]])]
    normaliser = GlobalNormalizer(2, 3)
    try:
        normaliser.transform(X)
    except Exception as e:
        target = (
            "This instance of GlobalNormalizer has not "
            "been fitted yet; please call ``fit`` first."
        )
        assert str(e) == target

    X_transformed = normaliser.fit_transform(X)

    mean = np.mean([np.mean(x) for x in X_transformed])
    std = np.std([np.std(x) for x in X_transformed])
    assert np.allclose(mean, 2)
    assert np.allclose(std, 3)

    X_inv = normaliser.inverse_transform(X_transformed)
    assert np.all([np.allclose(x, x_inv) for x, x_inv in zip(X, X_inv)])


def test_global_centering():
    """Test the GlobalCenterer class."""
    X = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9, 10], [10, 11, 12, 13]])]
    centerer = GlobalCenterer(2)
    try:
        centerer.transform(X)
    except Exception as e:
        target = (
            "This instance of GlobalCenterer has not "
            "been fitted yet; please call ``fit`` first."
        )
        assert str(e) == target

    X_transformed = centerer.fit_transform(X)

    mean = np.mean([np.mean(x) for x in X_transformed])
    assert np.allclose(mean, 2)

    X_inv = centerer.inverse_transform(X_transformed)
    assert np.all([np.allclose(x, x_inv) for x, x_inv in zip(X, X_inv)])


def test_global_min_max():
    """Test the GlobalMinMaxScaler class."""
    X = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9, 10], [10, 11, 12, 13]])]
    minmax = GlobalMinMaxScaler(-2, 2)
    try:
        minmax.transform(X)
    except Exception as e:
        target = (
            "This instance of GlobalMinMaxScaler has not "
            "been fitted yet; please call ``fit`` first."
        )
        assert str(e) == target

    X_transformed = minmax.fit_transform(X)

    min_val = np.min([np.min(x) for x in X_transformed])
    max_val = np.max([np.max(x) for x in X_transformed])

    assert np.allclose(min_val, -2)
    assert np.allclose(max_val, 2)

    X_inv = minmax.inverse_transform(X_transformed)
    assert np.all([np.allclose(x, x_inv) for x, x_inv in zip(X, X_inv)])
