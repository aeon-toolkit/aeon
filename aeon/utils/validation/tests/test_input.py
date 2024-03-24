"""Test functions for input validation."""

import pytest

from aeon.testing.utils.data_gen import get_examples
from aeon.testing.utils.data_gen._collection import EQUAL_LENGTH_UNIVARIATE
from aeon.utils.validation._input import is_hierarchical, is_valid_input, validate_input
from aeon.utils.validation.collection import is_collection
from aeon.utils.validation.series import is_single_series

# Tests limited to these input, some  collection types are ambiguous
COLLECTIONS = ["numpy3D", "nested_univ", "pd-multiindex"]
SERIES = ["pd.Series", "pd.DataFrame", "np.ndarray"]
HIERARCHICAL = ["pd_multiindex_hier"]
PANDAS_TYPES = ["pd.Series", "pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


@pytest.mark.parametrize("data", COLLECTIONS)
def test_input_collections(data):
    """Test is_collection with correct input."""
    assert is_collection(EQUAL_LENGTH_UNIVARIATE[data])
    assert not is_single_series(EQUAL_LENGTH_UNIVARIATE[data])
    assert not is_hierarchical(EQUAL_LENGTH_UNIVARIATE[data])


@pytest.mark.parametrize("data_type", HIERARCHICAL)
def test_input_hierarchy(data_type):
    """Test validation checks for hierarchical input."""
    data = get_examples(data_type)
    for d in data:
        assert not is_collection(d)
        assert not is_single_series(d)
        assert is_hierarchical(d)
        b, m = validate_input(d)
        assert b
        assert m["scitype"] == "Hierarchical"
        assert m["mtype"] == data_type
        b, m = validate_input("A string")
        assert not b
        assert m is None


@pytest.mark.parametrize("data_type", SERIES)
def test_input_series(data_type):
    """Test  validation checks for single series."""
    data = get_examples(data_type)
    for d in data:
        assert not is_collection(d)
        assert is_single_series(d)
        assert not is_hierarchical(d)
        b, m = validate_input(d)
        assert b
        assert m["scitype"] == "Series"
        assert m["mtype"] == data_type


@pytest.mark.parametrize("data_type", COLLECTIONS)
def test_input_collection(data_type):
    """Test is_collection with correct input."""
    d = EQUAL_LENGTH_UNIVARIATE[data_type]
    assert is_collection(d)
    assert not is_single_series(d)
    assert not is_hierarchical(d)
    b, m = validate_input(d)
    assert b
    assert m["scitype"] == "Panel"
    assert m["mtype"] == data_type


@pytest.mark.parametrize("panda_type", PANDAS_TYPES)
def test_pandas_valid_input(panda_type):
    """Test is_valid_input."""
    res = get_examples(panda_type)
    for r in res:
        assert is_valid_input(r)
