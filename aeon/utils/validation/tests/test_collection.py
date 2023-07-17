# -*- coding: utf-8 -*-
"""Unit tests for aeon.utils.validation.collection check/convert functions."""
import numpy as np
import pandas as pd
import pytest

from aeon.datasets import make_example_multi_index_dataframe

# from aeon.datasets._data_generators import make_example_multi_index_dataframe
from aeon.utils._testing.tests.test_collection import make_nested_dataframe_data
from aeon.utils.validation.collection import (  # _nested_uni_is_equal,; has_missing,
    COLLECTIONS_DATA_TYPES,
    _is_nested_univ_dataframe,
    convertX,
    equal_length,
    get_n_cases,
    get_type,
)

np_list = []
for _ in range(10):
    np_list.append(np.random.random(size=(2, 20)))
df_list = []
for _ in range(10):
    df_list.append(pd.DataFrame(np.random.random(size=(2, 20))))
nested, _ = make_nested_dataframe_data()
multi = make_example_multi_index_dataframe()

EQUAL_LENGTH_DATA_EXAMPLES = {
    "numpy3D": np.random.random(size=(10, 3, 20)),
    "np-list": np_list,
    "df-list": df_list,
    "numpyflat": np.zeros(shape=(10, 20)),
    "pd-wide": pd.DataFrame(np.zeros(shape=(10, 20))),
    "nested_univ": nested,
    "pd-multiindex": multi,
}
np_list_uneq = []
for i in range(10):
    np_list_uneq.append(np.random.random(size=(2, 20 + i)))
df_list_uneq = []
for i in range(10):
    df_list_uneq.append(pd.DataFrame(np.random.random(size=(2, 20 + i))))

UNEQUAL_DATA_EXAMPLES = {
    "np-list": np_list_uneq,
    "df-list": df_list_uneq,
}


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_equal_length(data):
    assert equal_length(EQUAL_LENGTH_DATA_EXAMPLES[data], data)


@pytest.mark.parametrize("data", ["df-list"])
def test_unequal_length(data):
    assert not equal_length(UNEQUAL_DATA_EXAMPLES[data], data)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_type(data):
    assert get_type(EQUAL_LENGTH_DATA_EXAMPLES[data]) == data


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test__is_nested_univ_dataframe(data):
    if data == "nested_univ":
        assert _is_nested_univ_dataframe(EQUAL_LENGTH_DATA_EXAMPLES[data])
    else:
        assert not _is_nested_univ_dataframe(EQUAL_LENGTH_DATA_EXAMPLES[data])


@pytest.mark.parametrize("input_type", COLLECTIONS_DATA_TYPES)
@pytest.mark.parametrize("output_type", COLLECTIONS_DATA_TYPES)
def test_convertX_equal_length(input_type, output_type):
    # dont test conversion from unequal supporting to equal only, or multivariate to
    input = EQUAL_LENGTH_DATA_EXAMPLES[input_type]
    n_cases = get_n_cases(input)
    result = convertX(input, output_type)
    n_cases2 = get_n_cases(result)
    t = get_type(result)
    assert t == output_type
    assert n_cases == n_cases2
