"""Test for SAX transformations on time series."""

import numpy as np
import pytest

from aeon.transformations.collection.dictionary_based import SAX


@pytest.mark.parametrize("n_segments", [8])
@pytest.mark.parametrize("alphabet_size", [4])
def test_equal_length_univariate_sax(n_segments, alphabet_size):
    """Test if SAX maintains equal length for univariate time series.

    Uses both transform and the inverse transformation.
    """
    X = np.random.normal(size=(10, 1, 100))

    # normalize input
    stds = np.std(X, axis=2, keepdims=True)
    stds[stds == 0] = 1.0
    X = (X - np.mean(X, axis=2, keepdims=True)) / stds

    sax = SAX(n_segments=n_segments, alphabet_size=alphabet_size)

    X_sax = sax.fit_transform(X=X)
    X_sax_inv = sax.inverse_sax(X=X_sax, original_length=100)

    assert X_sax.shape[-1] == n_segments
    assert X_sax_inv.shape[-1] == X.shape[-1]
    assert len(sax.breakpoints) == 3
    assert len(sax.breakpoints_mid) == 4


def test_sax_unsupported_distribution_raises():
    """Test an unsupported distribution raises a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        SAX(distribution="bogus")


def test_sax_znormalized_false_normalizes_internally():
    """Test znormalized=False applies normalization before the PAA step."""
    X = np.random.RandomState(0).normal(loc=5, scale=2, size=(5, 1, 40))
    sax = SAX(n_segments=8, alphabet_size=4, znormalized=False)
    X_sax = sax.fit_transform(X=X)
    assert X_sax.shape[-1] == 8


def test_sax_get_test_params():
    """Test the default test parameters are valid and usable."""
    params = SAX._get_test_params()
    sax = SAX(**params)
    assert isinstance(sax, SAX)


@pytest.mark.parametrize("n_segments", [8])
@pytest.mark.parametrize("alphabet_size", [4])
def test_equal_length_multivariate_sax(n_segments, alphabet_size):
    """Test if SAX maintains equal length for multivariate time series.

    Uses both transform and the inverse transformation.
    """
    X = np.random.normal(size=(10, 3, 100))

    # normalize input
    stds = np.std(X, axis=2, keepdims=True)
    stds[stds == 0] = 1.0
    X = (X - np.mean(X, axis=2, keepdims=True)) / stds

    sax = SAX(n_segments=n_segments, alphabet_size=alphabet_size)

    X_sax = sax.fit_transform(X=X)
    X_sax_inv = sax.inverse_sax(X=X_sax, original_length=100)

    assert X_sax.shape[-1] == n_segments
    assert X_sax_inv.shape[-1] == X.shape[-1]
    assert len(sax.breakpoints) == 3
    assert len(sax.breakpoints_mid) == 4
