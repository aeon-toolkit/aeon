# -*- coding: utf-8 -*-
"""Unit tests for aeon.utils.validation.collection check/convert functions."""
import numpy as np
import pandas as pd
import pytest

from aeon.datasets import make_example_multi_index_dataframe

# from aeon.datasets._data_generators import make_example_multi_index_dataframe
from aeon.utils._testing.tests.test_collection import make_nested_dataframe_data
from aeon.utils.validation.collection import (  # _nested_uni_is_equal,; ,
    COLLECTIONS_DATA_TYPES,
    _is_nested_univ_dataframe,
    equal_length,
    get_n_cases,
    get_type,
    has_missing,
    is_equal_length,
)

np_list = []
for _ in range(10):
    np_list.append(np.random.random(size=(1, 20)))
df_list = []
for _ in range(10):
    df_list.append(pd.DataFrame(np.random.random(size=(1, 20))))
nested, _ = make_nested_dataframe_data(n_cases=10)
multi = make_example_multi_index_dataframe(n_instances=10)

EQUAL_LENGTH_UNIVARIATE = {
    "numpy3D": np.random.random(size=(10, 1, 20)),
    "np-list": np_list,
    "df-list": df_list,
    "numpyflat": np.zeros(shape=(10, 20)),
    "pd-wide": pd.DataFrame(np.zeros(shape=(10, 20))),
    "nested_univ": nested,
    "pd-multiindex": multi,
}
np_list_uneq = []
for i in range(10):
    np_list_uneq.append(np.random.random(size=(1, 20 + i)))
df_list_uneq = []
for i in range(10):
    df_list_uneq.append(pd.DataFrame(np.random.random(size=(1, 20 + i))))

nested_univ_uneq = pd.DataFrame(dtype=np.float32)
instance_list = []
for i in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20 + i)))
nested_univ_uneq["channel0"] = instance_list

UNEQUAL_LENGTH_DATA_EXAMPLES = {
    "np-list": np_list_uneq,
    "df-list": df_list_uneq,
    "nested_univ": nested_univ_uneq,
}


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_n_cases(data):
    """Test getting the number of cases."""
    assert get_n_cases(EQUAL_LENGTH_UNIVARIATE[data]) == 10


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_type(data):
    """Test getting the type."""
    assert get_type(EQUAL_LENGTH_UNIVARIATE[data]) == data


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_equal_length(data):
    """Test if equal length series correctly identified."""
    assert equal_length(EQUAL_LENGTH_UNIVARIATE[data], data)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_equal_length(data):
    """Test if equal length series correctly identified."""
    assert is_equal_length(EQUAL_LENGTH_UNIVARIATE[data])


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not equal_length(UNEQUAL_LENGTH_DATA_EXAMPLES[data], data)


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_is_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not is_equal_length(UNEQUAL_LENGTH_DATA_EXAMPLES[data])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_has_missing(data):
    assert not has_missing(EQUAL_LENGTH_UNIVARIATE[data])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test__is_nested_univ_dataframe(data):
    if data == "nested_univ":
        assert _is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not _is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])
