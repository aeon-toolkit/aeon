"""Test forecasting generators."""

import pandas as pd
import pytest

from aeon.testing.data_generation.forecasting import (
    _assert_correct_columns,
    _get_n_columns,
)


def test_get_n_columns():
    """Test get_n_columns."""
    # Test for the 'univariate' tag
    assert _get_n_columns("univariate") == [1, 2], "Failed on 'univariate' tag"
    # Test for the 'multivariate' tag
    assert _get_n_columns("multivariate") == [2], "Failed on 'multivariate' tag"
    # Test for the 'both' tag
    assert _get_n_columns("both") == [1, 2], "Failed on 'both' tag"
    # Test for an unexpected tag
    with pytest.raises(ValueError):
        _get_n_columns("unknown")


def test_assert_correct_columns():
    """Test assert correct columns."""
    # Example setup with DataFrame
    y_train_df = pd.DataFrame([[1, 2]], columns=["A", "B"])
    y_pred_df = pd.DataFrame([[3, 4]], columns=["A", "B"])
    _assert_correct_columns(y_pred_df, y_train_df)  # Should pass without error

    # Example setup with Series
    y_train_series = pd.Series([1, 2], name="A")
    y_pred_series = pd.Series([3, 4], name="A")
    _assert_correct_columns(y_pred_series, y_train_series)  # Should pass without error

    # Failure case for DataFrame
    y_pred_df_wrong = pd.DataFrame([[3, 4]], columns=["X", "Y"])
    with pytest.raises(AssertionError):
        _assert_correct_columns(y_pred_df_wrong, y_train_df)

    # Failure case for Series
    y_pred_series_wrong = pd.Series([3, 4], name="X")
    with pytest.raises(AssertionError):
        _assert_correct_columns(y_pred_series_wrong, y_train_series)
