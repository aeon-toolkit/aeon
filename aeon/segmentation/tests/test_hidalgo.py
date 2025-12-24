"""Test Hidalgo segmenter."""

import numpy as np
import pytest

from aeon.segmentation._hidalgo import HidalgoSegmenter, _binom, _partition_function


def test_partition_function():
    """Test Hidalgo segmenter partition function."""
    p = _partition_function(10, 2, 0, 1)
    assert p == 8.0
    b = _binom(10, 2)
    assert b == 45.0


def test_hidalgo_empty_sampling_error():
    """Test Hidalgo Segmenter to raise ValueError when no samples are collected."""
    X = np.random.rand(50, 3)
    model = HidalgoSegmenter(K=2, n_iter=10, sampling_rate=10, burn_in=0.9, seed=42)

    with pytest.raises(ValueError, match="HidalgoSegmenter failed to collect"):
        model.fit(X)


def test_hidalgo_small_input_error():
    """Test that HidalgoSegmenter raises ValueError for small input size."""
    X = np.random.rand(3, 3)
    model = HidalgoSegmenter(q=3)

    with pytest.raises(ValueError, match="is too small for q="):
        model.fit(X)
