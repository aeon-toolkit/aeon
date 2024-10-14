"""Test summary features transformer."""

import pytest

from aeon.transformations.collection.feature_based import SevenNumberSummaryTransformer


def test_summary_features():
    """Test get functions."""
    x = SevenNumberSummaryTransformer()
    f = x._get_functions()
    assert len(f) == 7
    assert callable(f[0])
    x = SevenNumberSummaryTransformer(summary_stats="percentiles")
    assert len(x._get_functions()) == 7
    assert isinstance(x[0], float)
    assert x[1] == 0.887
    x = SevenNumberSummaryTransformer(summary_stats="bowley")
    assert len(x._get_functions()) == 7
    assert callable(x[0])
    assert x[6] == 0.9
    x = SevenNumberSummaryTransformer(summary_stats="tukey")
    assert len(x._get_functions()) == 7
    with pytest.raises(ValueError, match="Invalid summary_stats"):
        x = SevenNumberSummaryTransformer(summary_stats="invalid")
