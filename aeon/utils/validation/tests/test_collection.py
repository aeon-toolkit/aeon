#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Unit tests for aeon.utils.validation.collection check/convert functions."""
import numpy as np
import pandas as pd
import pytest

# from aeon.datasets._data_generators import make_example_multi_index_dataframe
from aeon.utils._testing.tests.test_collection import make_nested_dataframe_data
from aeon.utils.validation.collection import (  # _nested_uni_is_equal,; has_missing,
    DATA_TYPES,
    _is_nested_dataframe,
    convertX,
    equal_length,
    get_type,
)

np_list = []
for _ in range(10):
    np_list.append(np.zeros(shape=(20, 2)))
df_list = []
for _ in range(10):
    df_list.append(pd.DataFrame(np.zeros(shape=(20, 2))))
nested, _ = make_nested_dataframe_data()
# multi = make_example_multi_index_dataframe()

DATA_EXAMPLES = {
    "numpy3D": np.zeros(shape=(10, 3, 20)),
    "numpyflat": np.zeros(shape=(10, 20)),
    "np-list": np_list,
    "df-list": df_list,
    "pd-wide": pd.DataFrame(np.zeros(shape=(10, 20))),
    "nested_univ": nested,
}
#    "pd-multiindex": multi,


@pytest.mark.parametrize("data", DATA_TYPES)
def test_equal_length(data):
    assert equal_length(DATA_EXAMPLES[data], data)


@pytest.mark.parametrize("data", DATA_TYPES)
def test_get_type(data):
    assert get_type(DATA_EXAMPLES[data]) == data


@pytest.mark.parametrize("data", DATA_TYPES)
def test_is_nested_dataframe(data):
    if data == "nested_univ":
        assert _is_nested_dataframe(DATA_EXAMPLES[data])
    else:
        assert not _is_nested_dataframe(DATA_EXAMPLES[data])


@pytest.mark.parametrize("input_data", DATA_TYPES)
@pytest.mark.parametrize("output_data", DATA_TYPES)
def test_convertX(input_data, output_data):
    # dont test conversion from unequal supporting to equal only, or multivariate to
    # univariate only. pd-wide seems unsupported.
    X = convertX(DATA_EXAMPLES[input_data], output_data)
    t = get_type(X)
    assert t == output_data
