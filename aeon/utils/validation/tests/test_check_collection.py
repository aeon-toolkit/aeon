"""Test check collection functionality."""

import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_example_nested_dataframe
from aeon.testing.utils.data_gen._collection import EQUAL_LENGTH_UNIVARIATE
from aeon.utils.conversion._convert_collection import COLLECTIONS_DATA_TYPES
from aeon.utils.validation._check_collection import (
    _is_pd_wide,
    _nested_univ_is_equal,
    is_nested_univ_dataframe,
)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test__is_nested_univ_dataframe(data):
    if data == "nested_univ":
        assert is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])


def test__nested_univ_is_equal():
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
def test__is_pd_wide(data):
    if data == "pd-wide":
        assert _is_pd_wide(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not _is_pd_wide(EQUAL_LENGTH_UNIVARIATE[data])
