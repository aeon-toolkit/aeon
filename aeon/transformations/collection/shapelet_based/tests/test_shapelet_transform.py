"""Tests for shapelet transform functions."""

__maintainer__ = []

from aeon.transformations.collection.shapelet_based._shapelet_transform import (
    _calc_binary_ig,
)


def test_calc_binary_ig_identical_splits():
    """Test binary IG calculation correctly handles identical split values.

    Regression test for Issue #1322. Ensures that split points are not
    evaluated between identical distance values, which previously resulted
    in incorrect Information Gain calculation.
    """
    orderline = [(2, -1), (2, -1), (2, 1), (3, 1), (3, 1)]

    # Class 1 has 3 items, Class -1 has 2 items
    c1, c2 = 3, 2

    ig = _calc_binary_ig(orderline, c1, c2)

    # The expected IG is approx 0.42.
    assert 0.41 < ig < 0.43
