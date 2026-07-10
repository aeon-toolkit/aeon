"""Test for PAA transformations on time series."""

import numpy as np
import pytest

from aeon.transformations.collection.dictionary_based import PAA


@pytest.mark.parametrize("n_segments", [8])
def test_equal_length_univariate_paa(n_segments):
    """Test if PAA maintains equal length for univariate time series.

    Uses both transform and the inverse transformation.
    """
    X = np.random.normal(size=(10, 1, 100))

    # normalize input
    stds = np.std(X, axis=2, keepdims=True)
    stds[stds == 0] = 1.0
    X = (X - np.mean(X, axis=2, keepdims=True)) / stds

    paa = PAA(n_segments=n_segments)

    X_paa = paa.fit_transform(X=X)
    X_paa_inv = paa.inverse_paa(X=X_paa, original_length=100)

    assert X_paa.shape[-1] == n_segments
    assert X_paa_inv.shape[-1] == X.shape[-1]


def test_paa_length_divisible_by_n_segments():
    """Test the fast path when series length is divisible by n_segments."""
    X = np.random.normal(size=(5, 1, 100))
    paa = PAA(n_segments=10)

    X_paa = paa.fit_transform(X=X)
    X_paa_inv = paa.inverse_paa(X=X_paa, original_length=100)

    assert X_paa.shape[-1] == 10
    assert X_paa_inv.shape == X.shape


def test_paa_more_segments_than_timepoints():
    """Test PAA with more segments than timepoints produces empty-segment zeros."""
    X = np.random.normal(size=(3, 1, 5))
    paa = PAA(n_segments=8)

    X_paa = paa.fit_transform(X=X)

    assert X_paa.shape[-1] == 8
    # the trailing segments have no source timepoints and stay at zero
    np.testing.assert_array_equal(X_paa[:, :, 5:], np.zeros((3, 1, 3)))


def test_paa_get_test_params():
    """Test the default test parameters are valid and usable."""
    params = PAA._get_test_params()
    paa = PAA(**params)
    assert isinstance(paa, PAA)


@pytest.mark.parametrize("n_segments", [8])
def test_equal_length_multivariate_paa(n_segments):
    """Test if PAA maintains equal length for multivariate time series.

    Uses both transform and the inverse transformation.
    """
    X = np.random.normal(size=(10, 3, 100))

    # normalize input
    stds = np.std(X, axis=2, keepdims=True)
    stds[stds == 0] = 1.0
    X = (X - np.mean(X, axis=2, keepdims=True)) / stds

    paa = PAA(n_segments=n_segments)

    X_paa = paa.fit_transform(X=X)
    X_paa_inv = paa.inverse_paa(X=X_paa, original_length=100)

    assert X_paa.shape[-1] == n_segments
    assert X_paa_inv.shape[-1] == X.shape[-1]
