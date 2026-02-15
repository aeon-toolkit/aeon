"""Test Hidalgo segmenter."""

import numpy as np

from aeon.segmentation._hidalgo import HidalgoSegmenter, _binom, _partition_function


def test_partition_function():
    """Test Hidalgo segmenter partition function."""
    p = _partition_function(10, 2, 0, 1)
    assert p == 8.0
    b = _binom(10, 2)
    assert b == 45.0


def test_hidalgo_zero_distance_duplicate_rows():
    """Test that Hidalgo handles duplicate rows without numerical errors."""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    seg = HidalgoSegmenter(
        K=2,
        q=2,
        n_iter=50,
        burn_in=0.2,
        sampling_rate=5,
        seed=1,
    )
    out = seg.fit_predict(X, axis=0)
    assert out is not None
    assert isinstance(out, np.ndarray)
    assert len(out) == len(X)
