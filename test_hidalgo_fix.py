"""Tests for HidalgoSegmenter empty sampling fix.

Tests verify that HidalgoSegmenter handles edge cases when Gibbs samples
don't survive burn-in and sampling_rate filtering.
"""

import numpy as np
import pytest

from aeon.segmentation import HidalgoSegmenter


def test_hidalgo_empty_sampling_raises_valueerror():
    """Test that empty sampling after filtering raises informative error."""
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    seg = HidalgoSegmenter(K=2, q=2, n_iter=10, burn_in=0.9, sampling_rate=10, seed=1)
    with pytest.raises(ValueError, match="No valid samples"):
        seg.fit(X)


def test_hidalgo_valid_configuration():
    """Test that valid configuration produces valid results."""
    X = np.random.default_rng(42).standard_normal((50, 4))
    seg = HidalgoSegmenter(K=2, q=2, n_iter=50, burn_in=0.3, sampling_rate=2, seed=42)
    result = seg.fit_predict(X, axis=0)
    assert result is not None
    assert len(result) == 50
    assert np.all((result >= -1) & (result < 2))


def test_hidalgo_boundary_sampling():
    """Test boundary conditions for sampling parameters."""
    X = np.random.default_rng(42).standard_normal((30, 3))
    seg = HidalgoSegmenter(K=2, q=2, n_iter=100, burn_in=0.5, sampling_rate=5, seed=42)
    result = seg.fit_predict(X, axis=0)
    assert result is not None
    assert len(result) == 30
