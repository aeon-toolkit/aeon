# -*- coding: utf-8 -*-
"""Unit tests for collection converters."""
import numpy as np
import pandas as pd
import pytest

from aeon.utils.validation.collection import DATA_TYPES, convert_dict
from aeon.utils.validation.tests.test_collection import DATA_EXAMPLES


def type_match(input, output_type):
    if output_type == "numpy3D":
        return isinstance(input, np.ndarray) and input.ndim == 3
    if output_type == "np-list":
        return isinstance(input, list) and isinstance(input[0], np.ndarray)
    if output_type == "df-list":
        return isinstance(input, list) and isinstance(input[0], pd.DataFrame)
    if output_type == "numpyflat":
        return isinstance(input, np.ndarray) and input.ndim == 2
    if output_type == "pd-wide":
        return isinstance(input, pd.DataFrame)
    if output_type == "nested_univ":
        return isinstance(input, pd.DataFrame) and isinstance(
            input, input.iloc[0, 0], pd.Series
        )
    raise TypeError(f"Unknown collection output type string {output_type}")


@pytest.mark.parametrize("input_type", DATA_TYPES)
@pytest.mark.parametrize("output_type", DATA_TYPES)
def test_collection_converters(input_type, output_type):
    input = DATA_EXAMPLES(input_type)
    converter = convert_dict(input_type, output_type)
    result = converter(input)
    assert type_match(type(result), output_type)
