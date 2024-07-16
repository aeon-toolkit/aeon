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
