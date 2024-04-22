"""Test check collection functionality."""

import numpy as np
import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_example_nested_dataframe
from aeon.testing.utils.data_gen._collection import EQUAL_LENGTH_UNIVARIATE
from aeon.utils import COLLECTIONS_DATA_TYPES
from aeon.utils.validation.collection import (
    _is_pd_wide,
    _nested_univ_is_equal,
    get_type,
    has_missing,
    is_nested_univ_dataframe,
    is_tabular,
)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_nested_univ_dataframe(data):
    """Test is_nested_univ_dataframe function for different datatypes."""
    if data == "nested_univ":
        assert is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])


def test_nested_univ_is_equal():
    """Test _nested_univ_is_equal function for pd.DataFrame.

    Note that the function _nested_univ_is_equal assumes series are equal length
    over channels so only tests the first channel.
    """
    data = {
        "A": [pd.Series([1, 2, 3, 4]), pd.Series([4, 5, 6])],
        "B": [pd.Series([1, 2, 3, 4]), pd.Series([4, 5, 6])],
        "C": [pd.Series([1, 2, 3, 4]), pd.Series([4, 5, 6])],
    }
    X = pd.DataFrame(data)
    assert not _nested_univ_is_equal(X)
    X, _ = make_example_nested_dataframe(n_cases=10, n_channels=1, n_timepoints=20)
    assert _nested_univ_is_equal(X)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_pd_wide(data):
    """Test _is_pd_wide function for different datatypes."""
    if data == "pd-wide":
        assert _is_pd_wide(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not _is_pd_wide(EQUAL_LENGTH_UNIVARIATE[data])


def test_is_tabular():
    """Test is_tabular function."""
    d1 = np.random.random(size=(10, 10))
    assert is_tabular(d1)
    assert is_tabular(pd.DataFrame(d1))
    assert not is_tabular(np.random.random(size=(10, 10, 10)))
    assert not is_tabular(np.random.random(size=(10)))


def test_get_type():
    """Test get_type function."""
    np_list = [np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8])]
    with pytest.raises(TypeError, match="np-list must contain 2D np.ndarray"):
        get_type(np_list)
    dp_list = [pd.DataFrame(np.random.random(size=(10, 10))), np.array([1, 2, 3, 4, 5])]
    with pytest.raises(TypeError, match="df-list must only contain pd.DataFrame"):
        get_type(dp_list)
    res = get_type(pd.DataFrame(np.random.random(size=(10, 10))))
    assert res == "pd-wide"
    data = {
        "Double_Column": [1.5, 2.3, 3.6, 4.8, 5.2],
        "String_Column": ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(TypeError, match="contains non float values"):
        get_type(df)


def test_has_missing():
    """Test has_missing function."""
    d1 = np.random.random(size=(10, 10))
    assert not has_missing(d1)
    d1[0, 0] = np.nan
    assert has_missing(d1)
    d2 = np.random.random(size=(10, 10))
    l1 = [d1, d1]
    assert has_missing(l1)
    l2 = [pd.DataFrame(d1), pd.DataFrame(d2)]
    assert has_missing(l2)
