"""Unit tests for aeon.utils.validation.panel check functions."""

__author__ = ["mloning", "TonyBagnall"]
__all__ = ["test_check_X_bad_input_args"]

import numpy as np
import pandas as pd
import pytest

from aeon.utils._testing.collection import make_nested_dataframe_data
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
    """Test for the correct reaction for bad input in check_X.
    
    This test checks if the function check_X correctly raises a ValueError when it is given bad input. The bad input is defined in the list BAD_INPUT_ARGS and includes a list, a 1d np.array, a 4d np.array, and a non-nested pd.DataFrame. The test expects a ValueError to be raised when check_X is called with any of these inputs.
    """
    with pytest.raises(ValueError):
        check_X(X)

    with pytest.raises(ValueError):
        check_X_y(X, y)


def test_check_enforce_min_instances():
    """Test minimum instances enforced in check_X.
    
    This test checks if the function check_X correctly enforces a minimum number of instances. The test creates a nested DataFrame with 3 cases using the function make_nested_dataframe_data. It then calls check_X with enforce_min_instances set to 4, which is more than the number of cases in the DataFrame. The test expects a ValueError to be raised with the message 'instance'.
    """
    X, y = make_nested_dataframe_data(n_cases=3)
    msg = r"instance"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_min_instances=4)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_min_instances=4)

    with pytest.raises(ValueError, match=msg):
        check_y(y, enforce_min_instances=4)


def test_check_X_enforce_univariate():
    """Test univariate enforced in check_X.
    
    This test checks if the function check_X correctly enforces that the input is univariate. The test creates a nested DataFrame with 2 channels using the function make_nested_dataframe_data. It then calls check_X with enforce_univariate set to True. The test expects a ValueError to be raised with the message 'univariate'.
    """
    X, y = make_nested_dataframe_data(n_channels=2)
    msg = r"univariate"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_univariate=True)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_univariate=True)


def test_check_X_enforce_min_columns():
    """Test minimum columns enforced in check_X.
    
    This test checks if the function check_X correctly enforces a minimum number of columns. The test creates a nested DataFrame with 2 channels using the function make_nested_dataframe_data. It then calls check_X with enforce_min_columns set to 3, which is more than the number of channels in the DataFrame. The test expects a ValueError to be raised with the message 'columns'.
    """
    X, y = make_nested_dataframe_data(n_channels=2)
    msg = r"columns"
    with pytest.raises(ValueError, match=msg):
        check_X(X, enforce_min_columns=3)

    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, enforce_min_columns=3)


def test_check_X():
    """Test check X.
    
    This test checks various aspects of the function check_X. It first checks if check_X correctly raises a ValueError when both coerce_to_pandas and coerce_to_numpy are set to True. It then checks if check_X correctly converts a 2d np.array to a 3d np.array and enforces a minimum number of instances. Finally, it checks if check_X correctly converts a nested DataFrame to a np.ndarray when coerce_to_numpy is set to True.
    """
    with pytest.raises(ValueError, match="cannot both be set to True"):
        check_X(None, coerce_to_pandas=True, coerce_to_numpy=True)
    X = np.random.random(size=(5, 10))
    X2 = check_X(X)
    assert X2.shape == (5, 1, 10)
    check_X(X2, enforce_min_instances=5)
    with pytest.raises(ValueError, match="but a minimum of: 6 is required"):
        check_X(X2, enforce_min_instances=6)
    X, y = make_nested_dataframe_data(n_cases=5, n_channels=2, n_timepoints=10)
    X2 = check_X(X, coerce_to_numpy=True)
    assert isinstance(X2, np.ndarray)
    assert X2.shape == (5, 2, 10)


def test_check_y():
    """Test check y.
    
    This test checks various aspects of the function check_y. It first checks if check_y correctly raises a ValueError when it is given a string as input. It then checks if check_y correctly converts a pd.Series to a np.ndarray and enforces a minimum number of instances when coerce_to_numpy is set to True.
    """
    with pytest.raises(ValueError, match=" must be either a pd.Series or a np.ndarray"):
        check_y("Up the Arsenal")
    y = pd.Series([1, 2, 3, 4])
    y2 = check_y(y, coerce_to_numpy=True)
    assert isinstance(y2, np.ndarray)
    check_y(y2, enforce_min_instances=4)
    with pytest.raises(ValueError, match="but a minimum of: 5 is required"):
        check_y(y2, enforce_min_instances=5)
