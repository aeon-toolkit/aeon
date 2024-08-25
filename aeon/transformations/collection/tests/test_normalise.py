import numpy as np
import pytest

from aeon.transformations.collection.Normalise import Normalise


# Test function for z-normalization
def test_z_norm():
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalise(method="z_norm", axis=2)
    X_transformed = normaliser._transform(X)

    mean = np.mean(X_transformed, axis=2)
    std = np.std(X_transformed, axis=2)

    # Assert that the mean is close to 0 and standard deviation close to 1.
    assert np.allclose(mean, 0)
    assert np.allclose(std, 1)


# Test function for standardization
def test_standardize():
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalise(method="standardize", axis=2)
    X_transformed = normaliser._transform(X)

    mean = np.mean(X_transformed, axis=2)
    std = np.std(X_transformed, axis=2)

    # Assert that the mean is close to 0 and standard deviation close to 1.
    assert np.allclose(mean, 0)
    assert np.allclose(std, 1)


# Test function for min_max.
def test_min_max():
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalise(method="min_max", axis=2)
    X_transformed = normaliser._transform(X)

    min_val = np.min(X_transformed, axis=2)
    max_val = np.max(X_transformed, axis=2)

    # Assert that the min value is 0 and max value is 1.
    assert np.allclose(min_val, 0)
    assert np.allclose(max_val, 1)


def test_invalid_method():
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    with pytest.raises(ValueError, match="Unknown normalization method: invalid"):
        normaliser = Normalise(method="invalid", axis=2)
        normaliser._transform(X)
