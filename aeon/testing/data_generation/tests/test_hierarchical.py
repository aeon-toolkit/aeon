"""Test hierarchical generators."""

import pandas as pd

from aeon.testing.data_generation.hierarchical import _make_hierarchical


def test_make_hierarchical_basic():
    """Test make hierarchy."""
    df = _make_hierarchical()
    assert isinstance(df, pd.DataFrame), "Output is not a pandas DataFrame"
    assert df.shape[1] == 1, "DataFrame does not have the expected number of columns"
    assert not df.isnull().values.any(), "DataFrame contains unexpected NaN values"


def test_make_hierarchical_custom_levels():
    """Test make hierarchy."""
    # Test custom hierarchy levels
    hierarchy_levels = (3, 2)
    df = _make_hierarchical(hierarchy_levels=hierarchy_levels)
    expected_levels = len(hierarchy_levels) + 1  # +1 for the time index
    assert df.index.nlevels == expected_levels, "Incorrect number of index levels"


def test_make_hierarchical_timepoints_range():
    """Test make hierarchy."""
    # Test varying timepoints
    min_timepoints, max_timepoints = 5, 10
    df = _make_hierarchical(
        min_timepoints=min_timepoints, max_timepoints=max_timepoints, same_cutoff=False
    )
    # Verifying that series lengths vary within the specified range
    lengths = df.groupby(level=list(range(len(df.index.levels) - 1))).size()
    assert (
        lengths.min() >= min_timepoints and lengths.max() <= max_timepoints
    ), "Time points do not fall within the specified range"


def test_make_hierarchical_nan_values():
    """Test make hierarchy."""
    # Test NaN values inclusion
    df = _make_hierarchical(add_nan=True)
    assert df.isnull().values.any(), "DataFrame does not contain NaN values as expected"


def test_make_hierarchical_positive_values():
    """Test make hierarchy."""
    # Test all positive values
    df = _make_hierarchical(all_positive=True)
    assert (df >= 0).all().all(), "DataFrame contains non-positive values"


def test_make_hierarchical_index_type():
    """Test make hierarchy."""
    # Test for specific index types
    index_type = "datetime"
    df = _make_hierarchical(index_type=index_type)
    assert isinstance(
        df.index.get_level_values(-1)[0], pd.Timestamp
    ), "Index type does not match 'datetime'"
