"""Unit tests for aeon.utils.validation.panel check functions."""

__author__ = ["mloning", "TonyBagnall"]
__all__ = ["test_check_X_bad_input_args"]

import numpy as np
import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_example_nested_dataframe
from aeon.utils.validation.panel import check_X, check_X_y, check_y

BAD_INPUT_ARGS = [
    [0, 1, 2],  # list
    np.empty(2),  # 1d np.array
    np.empty((3, 2, 3, 2)),  # 4d np.array
    pd.DataFrame(np.empty((2, 3))),  # non-nested pd.DataFrame
]
y = pd.Series(dtype=int)


@pytest.mark.parametrize("X", BAD_INPUT_ARGS)
def test_check_X_bad_input_args(X):
    """Test for the correct reaction for bad input in check_X."""
    with pytest.raises(ValueError):
        check_X(X)

    with pytest.raises(ValueError):
        check_X_y(X, y)


def test_check_enforce_min_instances():
    """Test minimum instances enforced in check_X."""
    X, y = make_example_nested_dataframe(n_cases=3)
    msg = r"instance"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_min_instances=4)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_min_instances=4)

    with pytest.raises(ValueError, match=msg):
        check_y(y, enforce_min_instances=4)


def test_check_X_enforce_univariate():
    """Test univariate enforced in check_X."""
    X, y = make_example_nested_dataframe(n_channels=2)
    msg = r"univariate"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_univariate=True)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_univariate=True)


def test_check_X_enforce_min_columns():
    """Test minimum columns enforced in check_X."""
    X, y = make_example_nested_dataframe(n_channels=2)
    msg = r"columns"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_min_columns=3)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_min_columns=3)


def test_check_X():
    """Test check X."""
    with pytest.raises(ValueError, match="cannot both be set to True"):
        check_X(None, coerce_to_pandas=True, coerce_to_numpy=True)
    X = np.random.random(size=(5, 10))
    X2 = check_X(X)
    assert X2.shape == (5, 1, 10)
    check_X(X2, enforce_min_instances=5)
    with pytest.raises(ValueError, match="but a minimum of: 6 is required"):
        check_X(X2, enforce_min_instances=6)
    X, y = make_example_nested_dataframe(n_cases=5, n_channels=2, n_timepoints=10)
    X2 = check_X(X, coerce_to_numpy=True)
    assert isinstance(X2, np.ndarray)
    assert X2.shape == (5, 2, 10)


def test_check_y():
    """Test check y."""
    with pytest.raises(ValueError, match=" must be either a pd.Series or a np.ndarray"):
        check_y("Up the Arsenal")
    y = pd.Series([1, 2, 3, 4])
    y2 = check_y(y, coerce_to_numpy=True)
    assert isinstance(y2, np.ndarray)
    check_y(y2, enforce_min_instances=4)
    with pytest.raises(ValueError, match="but a minimum of: 5 is required"):
        check_y(y2, enforce_min_instances=5)
