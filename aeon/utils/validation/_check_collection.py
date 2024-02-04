"""Check collection utilities."""

import numpy as np
import pandas as pd


def is_nested_univ_dataframe(X):
    """Check if X is nested dataframe."""
    # Otherwise check all entries are pd.Series
    if not isinstance(X, pd.DataFrame):
        return False
    for _, series in X.items():
        for cell in series:
            if not isinstance(cell, pd.Series):
                return False
    return True


def _nested_univ_is_equal(X):
    """Check whether series in a nested DataFrame are of equal length.

    This function checks if all series in a nested DataFrame have the same length. It
    assumes that series are of equal length over channels, so it only tests the first
    channel.

    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame to check.

    Returns
    -------
    bool
        True if all series in the DataFrame are of equal length, False otherwise.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'A': [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])],
    ...     'B': [pd.Series([7, 8, 9]), pd.Series([10, 11, 12])]
    ... })
    >>> _nested_univ_is_equal(df)
    True
    """
    length = X.iloc[0, 0].size
    for i in range(1, X.shape[0]):
        if X.iloc[i, 0].size != length:
            return False
    return True


def _is_pd_wide(X):
    """Check whether the input nested DataFrame is "pd-wide" type."""
    # only test is if all values are float.
    if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.MultiIndex):
        if is_nested_univ_dataframe(X):
            return False
        float_cols = X.select_dtypes(include=[float]).columns
        for col in float_cols:
            if not np.issubdtype(X[col].dtype, np.floating):
                return False
        return True
    return False
