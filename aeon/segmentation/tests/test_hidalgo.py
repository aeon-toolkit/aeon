"""Test Hidalgo segmenter."""

from aeon.segmentation._hidalgo import _binom, _partition_function


def test_partition_function():
    """Test Hidalgo segmenter partition function."""
    p = _partition_function(10, 2, 0, 1)
    assert p == 8.0
    b = _binom(10, 2)
    assert b == 45.0
