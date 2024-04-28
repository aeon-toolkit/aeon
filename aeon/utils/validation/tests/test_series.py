"""Test series module."""

__maintainer__ = ["TonyBagnall"]

import numpy as np
import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_example_nested_dataframe
from aeon.utils.validation.series import (
    _check_is_multivariate,
    _check_pd_dataframe,
    _common_checks,
    check_consistent_index_type,
    check_equal_time_index,
    check_is_univariate,
    check_series,
    check_time_index,
    get_index_for_series,
    is_pdmultiindex_hierarchical,
    is_pred_interval_proba,
    is_pred_quantiles_proba,
    is_univariate_series,
)


def test_is_univariate_series():
    """Test Univariate Series."""
    assert not is_univariate_series(None)
    assert is_univariate_series(pd.Series([1, 2, 3, 4, 5]))
    assert is_univariate_series(np.array([1, 2, 3, 4, 5]))
    assert is_univariate_series(pd.DataFrame({"A": [1, 2, 3, 4, 5]}))
    assert not is_univariate_series(
        pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    )
    assert not is_univariate_series(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))


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
    """Test check_is_univariate."""
    X = np.random.random(size=(10, 1, 20))
    check_is_univariate(X)
    X = np.random.random(size=(10, 3, 20))
    with pytest.raises(ValueError, match="must be univariate"):
        check_is_univariate(X)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=1, n_timepoints=10)
    check_is_univariate(X)
    X, _ = make_example_nested_dataframe(n_cases=4, n_channels=2, n_timepoints=10)
    with pytest.raises(ValueError, match="must be univariate"):
        check_is_univariate(X)


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


def test_df_checks():
    """Test check_pd_dataframe function."""
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    df = pd.DataFrame(data)
    n1 = np.random.random(size=(10, 10))
    _check_pd_dataframe(df)
    assert _common_checks(df)
    assert not _common_checks(n1)
    assert not is_pred_interval_proba(n1)
    assert not is_pred_quantiles_proba(n1)
    assert not is_pdmultiindex_hierarchical(n1)
    assert isinstance(get_index_for_series(n1), pd.RangeIndex)
    assert isinstance(get_index_for_series(df), pd.RangeIndex)

    columns = ["A", "B", "B"]  # Notice the duplicate column name 'B'
    df = pd.DataFrame(data, columns=columns)
    assert not _common_checks(df)
    with pytest.raises(ValueError, match="must have unique column indices"):
        _check_pd_dataframe(df)

    index_strings = ["a", "b", "c"]
    # Creating the DataFrame
    df = pd.DataFrame(data, index=index_strings)
    with pytest.raises(ValueError, match="is not supported for series"):
        _check_pd_dataframe(df)
    data = {
        "Column1": [1, 2, {"a": 1, "b": 2}],
        # The third entry is a dictionary, which is an object
        "Column2": ["A", "B", "C"],
    }
    # Creating the DataFrame
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="should not have column of 'object' dtype"):
        _check_pd_dataframe(df)
    assert not _common_checks(df)
    index_non_monotonic = [1, 3, 2, 5, 4]  # This index is not monotonic increasing
    data = {"Column1": [10, 20, 30, 40, 50], "Column2": ["A", "B", "C", "D", "E"]}

    # Creating the DataFrame
    df = pd.DataFrame(data, index=index_non_monotonic)
    assert not _common_checks(df)


def test_is_pred_interval_proba():
    """Test is_pred_interval_proba."""
    # Create a correct MultiIndex DataFrame
    idx = pd.MultiIndex.from_tuples(
        [(1, 0.9, "upper"), (1, 0.9, "lower")], names=["level_0", "coverage", "bound"]
    )
    df_correct = pd.DataFrame([[0.1, 0.2]], columns=idx)

    # Create a DataFrame with incorrect MultiIndex levels
    idx_wrong_levels = pd.MultiIndex.from_tuples(
        [(1, "upper"), (1, "lower")], names=["level_0", "bound"]
    )
    df_wrong_levels = pd.DataFrame([[0.1, 0.2]], columns=idx_wrong_levels)

    # Create a DataFrame with incorrect data type in coverage level
    idx_wrong_dtype = pd.MultiIndex.from_tuples(
        [(1, "0.9", "upper"), (1, "0.9", "lower")],
        names=["level_0", "coverage", "bound"],
    )
    df_wrong_dtype = pd.DataFrame([[0.1, 0.2]], columns=idx_wrong_dtype)

    # Create a DataFrame with incorrect coverage values
    idx_wrong_values = pd.MultiIndex.from_tuples(
        [(1, 1.5, "upper"), (1, -0.1, "lower")], names=["level_0", "coverage", "bound"]
    )
    df_wrong_values = pd.DataFrame([[0.1, 0.2]], columns=idx_wrong_values)

    # Assertions
    assert is_pred_interval_proba(df_correct)
    assert not is_pred_interval_proba(df_wrong_levels)
    assert not is_pred_interval_proba(df_wrong_dtype)
    assert not is_pred_interval_proba(df_wrong_values)
