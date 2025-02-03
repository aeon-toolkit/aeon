"""Tests for deep_equals utility."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from aeon.testing.utils.deep_equals import deep_equals

# examples used for comparison below
DEEPEQUALS_ITEMS = [
    42,
    [],
    (()),
    [[[[()]]]],
    np.array([2, 3, 4]),
    np.array([2, 4, 5]),
    3.5,
    4.2,
    np.nan,
    pd.Series([1, 2], ["a", "b"]),
    pd.DataFrame({"a": [4, 2]}),
    pd.DataFrame({"a": [4, 3]}),
    (np.array([1, 2, 4]), [pd.DataFrame({"a": [4, 2]})]),
    {"foo": [42], "bar": pd.Series([1, 2])},
    {"bar": [12], "foo": pd.Series([1, 2])},
    csr_matrix([1, 2, 3]),
]
DEEPEQUALS_PAIRS = [
    (DEEPEQUALS_ITEMS[i], DEEPEQUALS_ITEMS[j])
    for i in range(len(DEEPEQUALS_ITEMS))
    for j in range(len(DEEPEQUALS_ITEMS))
    if i is not j
]


@pytest.mark.parametrize("item", DEEPEQUALS_ITEMS)
def test_deep_equals_positive(item):
    """Tests that deep_equals correctly identifies equal objects as equal."""
    x = deepcopy(item)
    y = deepcopy(item)
    eq, msg = deep_equals(x, y, return_msg=True)

    msg = (
        f"deep_equals incorrectly returned False for two identical copies of "
        f"the following object: {x}. msg = {msg}"
    )
    assert eq, msg


@pytest.mark.parametrize("item1, item2", DEEPEQUALS_PAIRS)
def test_deep_equals_negative(item1, item2):
    """Tests that deep_equals correctly identifies unequal objects as unequal."""
    x = deepcopy(item1)
    y = deepcopy(item2)
    eq = deep_equals(x, y)

    msg = (
        f"deep_equals incorrectly returned True when comparing "
        f"the following, different objects: x={x}, y={y}."
    )
    assert not eq, msg


def test_deep_equals_same():
    """Tests that deep_equals correctly identifies the same object as equal."""
    x = [1, 2, 3]
    eq = deep_equals(x, x)
    assert eq
