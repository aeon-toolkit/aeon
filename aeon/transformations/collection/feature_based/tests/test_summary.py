"""Test summary features transformer."""

import pytest

from aeon.transformations.collection.feature_based import SevenNumberSummary


def test_summary_features():
    """Test get functions."""
    x = SevenNumberSummary()
    f = x._get_functions()
    assert len(f) == 7
    assert callable(f[0])
    x = SevenNumberSummary(summary_stats="percentiles")
    f = x._get_functions()
    assert len(f) == 7
    assert isinstance(f[0], float)
    assert f[1] == 0.887
    x = SevenNumberSummary(summary_stats="bowley")
    f = x._get_functions()
    assert len(f) == 7
    assert callable(f[0])
    assert f[6] == 0.9
    x = SevenNumberSummary(summary_stats="tukey")
    assert len(x._get_functions()) == 7
    with pytest.raises(ValueError, match="Summary function input invalid"):
        x = SevenNumberSummary(summary_stats="invalid")
        x._get_functions()
