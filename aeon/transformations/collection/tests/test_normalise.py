"""
The module contains tests for the Normalise class.

It includes tests for different normalization methods such as
z-normalization, standardization, and min-max scaling. Additionally,
it tests the behavior of the Normalise class when provided with an
invalid normalization method.
"""

import numpy as np
import pytest

from aeon.transformations.collection._normalise import Normalise


# Test function for z-normalization
def test_z_norm():
    """
    Test the z-normalization method of the Normalise class.

    This function creates a 3D numpy array, applies
    z-normalization using the Normalise class, and asserts
    that the transformed data has a mean close to 0 and a
    standard deviation close to 1 along the specified axis.
    """
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalise(method="z_norm")
    X_transformed = normaliser._transform(X)

    mean = np.mean(X_transformed, axis=-1)
    std = np.std(X_transformed, axis=-1)

    assert np.allclose(mean, 0)
    assert np.allclose(std, 1)


# Test function for standardization
def test_standardize():
    """
    Test the standardization method of the Normalise class.

    This function creates a 3D numpy array, applies standardization
    using the Normalise class, and asserts that the transformed data
    has a mean close to 0 and a standard deviation close to 1 along
    the specified axis.
    """
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalise(method="standardize")
    X_transformed = normaliser._transform(X)

    mean = np.mean(X_transformed, axis=-1)

    assert np.allclose(mean, 0)


# Test function for min_max.
def test_min_max():
    """
    Test the min-max normalization method of the Normalise class.

    This function creates a 3D numpy array, applies min-max normalization
    using the Normalise class, and asserts that the transformed data has
    a minimum value of 0 and a maximum value of 1 along the specified axis.
    """
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalise(method="min_max")
    X_transformed = normaliser._transform(X)

    min_val = np.min(X_transformed, axis=-1)
    max_val = np.max(X_transformed, axis=-1)

    assert np.allclose(min_val, 0)
    assert np.allclose(max_val, 1)


def test_invalid_method():
    """
    Tests behavior of Normalise class when an invalid normalization method is provided.

    This function creates a 3D numpy array and attempts to apply an invalid
    normalization method using the Normalise class. It asserts that a ValueError
    is raised with the appropriate error message.
    """
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    with pytest.raises(ValueError, match="Unknown normalization method: invalid"):
        normaliser = Normalise(method="invalid")
        normaliser._transform(X)
