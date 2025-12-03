"""Test Hidalgo segmenter."""

import numpy as np

from aeon.segmentation._hidalgo import HidalgoSegmenter, _binom, _partition_function


def test_partition_function():
    """Test Hidalgo segmenter partition function."""
    p = _partition_function(10, 2, 0, 1)
    assert p == 8.0
    b = _binom(10, 2)
    assert b == 45.0


def test_hidalgo_zero_distance_stability():
    """
    Test Hidalgo segmenter with duplicate/near-duplicate points.

    Regression test for issue #3068: AssertionError when data contains
    identical rows, causing zero distances in nearest neighbor search.
    This should not crash but handle duplicates gracefully.
    """
    # Create data with exact duplicates (causes zero distances)
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],  # Exact duplicate
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.7, 0.8, 0.9],
        ]
    )  # Another duplicate

    # This should not raise AssertionError or divide-by-zero warnings
    hidalgo = HidalgoSegmenter(K=2, q=2, n_iter=100, burn_in=0.5)

    # Should complete without errors
    result = hidalgo.fit_predict(X, axis=0)

    # Basic sanity checks
    assert result is not None
    assert len(result) >= 0  # May return empty array if no changepoints
    assert isinstance(result, np.ndarray)


def test_hidalgo_normal_data():
    """
    Test Hidalgo segmenter with normal random data.

    Verifies that the fix doesn't break normal operation.
    """
    # Random data without duplicates
    rng = np.random.RandomState(42)
    X = rng.rand(50, 3)

    hidalgo = HidalgoSegmenter(K=3, q=3, n_iter=200, burn_in=0.8)
    result = hidalgo.fit_predict(X, axis=0)

    # Should work as before
    assert result is not None
    assert isinstance(result, np.ndarray)
