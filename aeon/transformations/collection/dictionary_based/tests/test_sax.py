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
    """Test znormalized=False z-normalizes each series internally before PAA.

    With ``znormalized=False`` SAX z-normalizes every series (to mean 0 and
    standard deviation 1) before the PAA step. Feeding raw, non-normalized data
    with ``znormalized=False`` must therefore (a) reproduce the result of
    z-normalizing the data by hand and using ``znormalized=True``, and
    (b) differ from skipping normalization altogether.
    """
    X = np.random.RandomState(0).normal(loc=5, scale=2, size=(5, 1, 40))

    # The same per-series z-normalization SAX applies internally.
    X_norm = (X - X.mean(axis=-1, keepdims=True)) / (
        X.std(axis=-1, keepdims=True) + 1e-8
    )
    assert np.allclose(X_norm.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(X_norm.std(axis=-1), 1.0, atol=1e-3)

    words_internal = SAX(
        n_segments=8, alphabet_size=4, znormalized=False
    ).fit_transform(X=X)
    words_prenorm = SAX(n_segments=8, alphabet_size=4, znormalized=True).fit_transform(
        X=X_norm
    )
    words_raw = SAX(n_segments=8, alphabet_size=4, znormalized=True).fit_transform(X=X)

    # Internal normalization reproduces the hand-normalized result ...
    np.testing.assert_array_equal(words_internal, words_prenorm)
    # ... and changes the output relative to leaving the data un-normalized.
    assert not np.array_equal(words_internal, words_raw)


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
