"""This module contains tests for the series module in the aeon.utils.validation package."""

__author__ = ["benheid", "TonyBagnall"]

import numpy as np
import pandas as pd
import pytest

from aeon.utils._testing.collection import make_nested_dataframe_data
from aeon.utils.validation.series import (
    _check_is_multivariate,
    _check_is_univariate,
    check_consistent_index_type,
    check_equal_time_index,
    check_series,
    check_time_index,
)


def test_check_equal_time_index():
    """Test the function check_equal_time_index.
    
    This function tests whether the function check_equal_time_index correctly checks if the time indices of two series are equal or contain the same values. It tests various scenarios, including when the time indices are equal, when they contain the same values but are not equal, and when they do not contain the same values.
    """
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
    """Test the function _check_is_univariate.
    
    This function tests whether the function _check_is_univariate correctly checks if a given series or DataFrame is univariate. It tests various scenarios, including when the series or DataFrame is univariate and when it is not.
    """
    X = np.random.random(size=(10, 1, 20))
    _check_is_univariate(X)
    X = np.random.random(size=(10, 3, 20))
    with pytest.raises(ValueError, match="must be univariate"):
        _check_is_univariate(X)
    X, _ = make_nested_dataframe_data(n_cases=4, n_channels=1, n_timepoints=10)
    _check_is_univariate(X)
    X, _ = make_nested_dataframe_data(n_cases=4, n_channels=2, n_timepoints=10)
    with pytest.raises(ValueError, match="must be univariate"):
        _check_is_univariate(X)


def test__check_is_multivariate():
    """Test the function _check_is_multivariate.
    
    This function tests whether the function _check_is_multivariate correctly checks if a given series or DataFrame is multivariate. It tests various scenarios, including when the series or DataFrame is multivariate and when it is not. It assumes that ndarrays are in (n_timepoints, n_channels) shape.
    """
    X = pd.Series([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match=" must have 2 or more variables, but found 1"):
        _check_is_multivariate(X)
    X, _ = make_nested_dataframe_data(n_cases=4, n_channels=2, n_timepoints=10)
    _check_is_multivariate(X)
    X, _ = make_nested_dataframe_data(n_cases=4, n_channels=1, n_timepoints=10)
    with pytest.raises(ValueError, match="must have 2 or more variables"):
        _check_is_multivariate(X)
    X = np.random.random(size=(10, 1))


def test_check_series():
    """Test the function check_series.
    
    This function tests whether the function check_series correctly checks if a given object is a series. It tests various scenarios, including when the object is a series and when it is not.
    """
    check_series(None)
    with pytest.raises(ValueError, match="cannot be None"):
        check_series(None, allow_None=False)
    X, _ = make_nested_dataframe_data(n_cases=4, n_channels=2, n_timepoints=10)
    with pytest.raises(ValueError, match="cannot both be set to True"):
        check_series(Z=X, enforce_univariate=True, enforce_multivariate=True)
    X, _ = make_nested_dataframe_data(n_cases=4, n_channels=2, n_timepoints=10)
    check_series(Z=X, enforce_multivariate=True)


def test_check_time_index():
    """Test the function check_time_index.
    
    This function tests whether the function check_time_index correctly checks if a given object is a valid time index. It tests various scenarios, including when the object is a valid time index and when it is not.
    """
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
    """Test the function check_consistent_index_type.
    
    This function tests whether the function check_consistent_index_type correctly checks if two given indices are of the same type. It tests various scenarios, including when the indices are of the same type and when they are not.
    """
    index1 = pd.RangeIndex(start=0, stop=5)
    index2 = pd.Index(["A", "B", "C", "D", "E"])

    # An exception should be raised because index types are inconsistent
    with pytest.raises(TypeError):
        check_consistent_index_type(index1, index2)
