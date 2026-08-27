"""Tests for signature method."""

import numpy as np
import pytest

from aeon.transformations.collection.signature_based import SignatureTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("roughpy", severity="none"),
    reason="skip test if required soft dependency roughpy not available",
)
def test_generalised_signature_method():
    """Check that dimension and dim of output are correct."""
    # Build an array X, note that this is [n_sample, n_channels, length] shape.
    from aeon.transformations.collection.signature_based._signature import (
        _sigdim,
    )

    n_channels = 3
    depth = 4
    X = np.random.randn(5, n_channels, 10)

    # Check the global dimension comes out correctly
    method = SignatureTransformer(depth=depth, window_name="global")
    assert method.fit_transform(X).shape[1] == _sigdim(n_channels + 1, depth) - 1

    # Check dyadic dim
    method = SignatureTransformer(depth=depth, window_name="dyadic", window_depth=3)
    assert method.fit_transform(X).shape[1] == (_sigdim(n_channels + 1, depth) - 1) * 15

    # Ensure an example
    X = np.array([[0, 1], [2, 3], [1, 1]]).reshape(-1, 2, 3)
    method = SignatureTransformer(depth=2, window_name="global")
    true_arr = np.array(
        [[1.0, 2.0, 1.0, 0.5, 1.33333333, -0.5, 0.66666667, 2.0, -1.0, 1.5, 3.0, 0.5]]
    )
    assert np.allclose(method.fit_transform(X), true_arr)


@pytest.mark.skipif(
    not _check_soft_dependencies("roughpy", severity="none"),
    reason="skip test if required soft dependency roughpy not available",
)
def test_window_error():
    """Test that wrong window parameters raise error."""
    X = np.random.randn(5, 2, 3)

    # Check dyadic gives a value error
    method = SignatureTransformer(window_name="dyadic", window_depth=10)
    with pytest.raises(ValueError):
        method.fit_transform(X)

    # Expanding and sliding errors
    method = SignatureTransformer(
        window_name="expanding", window_length=10, window_step=5
    )
    with pytest.raises(ValueError):
        method.fit_transform(X)
    method = SignatureTransformer(
        window_name="sliding", window_length=10, window_step=5
    )
    with pytest.raises(ValueError):
        method.fit_transform(X)


@pytest.mark.skipif(
    not _check_soft_dependencies("roughpy", severity="none"),
    reason="skip test if required soft dependency roughpy not available",
)
def test_depth_error():
    """Test that a non-positive signature depth raises an error."""
    X = np.random.randn(5, 2, 10)

    for depth in (0, -1):
        method = SignatureTransformer(depth=depth, window_name="global")
        with pytest.raises(ValueError, match="Depth must be at least 1"):
            method.fit_transform(X)
