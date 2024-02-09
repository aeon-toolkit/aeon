"""Test series module."""

__author__ = ["benheid", "TonyBagnall"]

import numpy as np
import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_example_nested_dataframe
from aeon.utils.validation.series import (
    _check_is_multivariate,
    _check_is_univariate,
    check_consistent_index_type,
    check_equal_time_index,
    check_series,
    check_time_index,
)


def test_check_equal_time_index():
    """Test check equal time index."""
    assert check_equal_time_index(None) is None
    x = (pd.Series([1, 2, 3, 4, 5]), pd.Series([2, 3, 4, 5, 6]))
    with pytest.raises(ValueError, match="mode must be "):
        check_equal_time_index(*x, mode="FOO")
    index1 = pd.date_range(start="2023-01-01", end="2023-01-05")
    index2 = pd.date_range(start="2023-01-06", end="2023-01-10")
    ys = (
        pd.Series([1, 2, 3, 4, 5], index=index1),
        pd.Series([6, 7, 8, 9, 10], index=index2),
    )
    with pytest.raises(ValueError):
        check_equal_time_index(*ys, mode="contains")
    with pytest.raises(ValueError):
        check_equal_time_index(*ys, mode="equal")


def test__check_is_univariate():
    """Test _check_is_univariate."""
    X = np.random.random(size=(10, 1, 20))
    _check_is_univariate(X)
    X = np.random.random(size=(10, 3, 20))
    with pytest.raises(ValueError, match="must be univariate"):
        _check_is_univariate(X)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=1, n_timepoints=10)
    _check_is_univariate(X)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=2, n_timepoints=10)
    with pytest.raises(ValueError, match="must be univariate"):
        _check_is_univariate(X)


def test__check_is_multivariate():
    """Test _check_is_multivariate.

    This function assumes ndarrays are in (n_timepoints, n_channels) shape.
    """
    X = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match=" must have 2 or more variables, but found 1"):
        _check_is_multivariate(X)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=2, n_timepoints=10)
    _check_is_multivariate(X)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=1, n_timepoints=10)
    with pytest.raises(ValueError, match="must have 2 or more variables"):
        _check_is_multivariate(X)
    X = np.random.random(size=(10, 1))


def test_check_series():
    """Test check_series."""
    check_series(None)
    with pytest.raises(ValueError, match="cannot be None"):
        check_series(None, allow_None=False)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=2, n_timepoints=10)
    with pytest.raises(ValueError, match="cannot both be set to True"):
        check_series(Z=X, enforce_univariate=True, enforce_multivariate=True)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=2, n_timepoints=10)
    check_series(Z=X, enforce_multivariate=True)


def test_check_time_index():
    """Test check_time_index."""
    x = np.array([1, 2, 3, 4, 5])
    with pytest.raises(NotImplementedError, match="is not supported"):
        check_time_index("HELLO")
    with pytest.raises(NotImplementedError, match="is not supported"):
        check_time_index(x, enforce_index_type=pd.Series)
    x = np.array([1, 2, 3, 5, 4])
    with pytest.raises(ValueError, match="must be sorted monotonically increasing"):
        check_time_index(x)
    x = pd.RangeIndex(0)
    with pytest.raises(ValueError, match="must contain at least some values"):
        check_time_index(x)


def test_check_consistent_index_type():
    """Test check_consistent_index_type."""
    index1 = pd.RangeIndex(start=0, stop=5)
    index2 = pd.Index(["A", "B", "C", "D", "E"])

    # An exception should be raised because index types are inconsistent
    with pytest.raises(TypeError):
        check_consistent_index_type(index1, index2)
