# -*- coding: utf-8 -*-
"""Unit tests for aeon.utils.validation.collection check/convert functions."""
import numpy as np
import pandas as pd
import pytest

from aeon.datasets import make_example_multi_index_dataframe
from aeon.utils._testing.tests.test_collection import make_nested_dataframe_data
from aeon.utils.validation._convert_collection import (
    _equal_length,
    _is_nested_univ_dataframe,
    _is_pd_wide,
)
from aeon.utils.validation.collection import (
    COLLECTIONS_DATA_TYPES,
    convert_collection,
    get_n_cases,
    get_type,
    has_missing,
    is_equal_length,
    is_univariate,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)

np_list = []
for _ in range(10):
    np_list.append(np.random.random(size=(1, 20)))
df_list = []
for _ in range(10):
    df_list.append(pd.DataFrame(np.random.random(size=(20, 1))))
nested, _ = make_nested_dataframe_data(n_cases=10)
multiindex = make_example_multi_index_dataframe(
    n_instances=10, n_channels=1, n_timepoints=20
)

EQUAL_LENGTH_UNIVARIATE = {
    "numpy3D": np.random.random(size=(10, 1, 20)),
    "np-list": np_list,
    "df-list": df_list,
    "numpyflat": np.zeros(shape=(10, 20)),
    "pd-wide": pd.DataFrame(np.zeros(shape=(10, 20))),
    "nested_univ": nested,
    "pd-multiindex": multiindex,
}
np_list_uneq = []
for i in range(10):
    np_list_uneq.append(np.random.random(size=(1, 20 + i)))
df_list_uneq = []
for i in range(10):
    df_list_uneq.append(pd.DataFrame(np.random.random(size=(20 + i, 1))))

nested_univ_uneq = pd.DataFrame(dtype=float)
instance_list = []
for i in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20 + i)))
nested_univ_uneq["channel0"] = instance_list

UNEQUAL_LENGTH_UNIVARIATE = {
    "np-list": np_list_uneq,
    "df-list": df_list_uneq,
    "nested_univ": nested_univ_uneq,
}
np_list_multi = []
for _ in range(10):
    np_list_multi.append(np.random.random(size=(2, 20)))
df_list_multi = []
for _ in range(10):
    df_list_multi.append(pd.DataFrame(np.random.random(size=(20, 2))))
multi = make_example_multi_index_dataframe(
    n_instances=10, n_channels=2, n_timepoints=20
)

nested_univ_multi = pd.DataFrame(dtype=float)
instance_list = []
for _ in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20)))
nested_univ_multi["channel0"] = instance_list
instance_list = []
for _ in range(0, 10):
    instance_list.append(pd.Series(np.random.randn(20)))
nested_univ_multi["channel1"] = instance_list


EQUAL_LENGTH_MULTIVARIATE = {
    "numpy3D": np.random.random(size=(10, 2, 20)),
    "np-list": np_list_multi,
    "df-list": df_list_multi,
    "nested_univ": nested_univ_multi,
    "pd-multiindex": multi,
}


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
@pytest.mark.parametrize("output_data", COLLECTIONS_DATA_TYPES)
def test_convert_collection(input_data, output_data):
    """Test all valid and invalid conversions."""
    # All should work with univariate equal length
    X = convert_collection(EQUAL_LENGTH_UNIVARIATE[input_data], output_data)
    assert get_type(X) == output_data
    # Test with multivariate
    if input_data in EQUAL_LENGTH_MULTIVARIATE:
        X = convert_collection(EQUAL_LENGTH_MULTIVARIATE[input_data], output_data)
        assert get_type(X) == output_data
    # Test with unequal length
    if input_data in UNEQUAL_LENGTH_UNIVARIATE:
        if output_data in UNEQUAL_LENGTH_UNIVARIATE or output_data == "pd-multiindex":
            X = convert_collection(UNEQUAL_LENGTH_UNIVARIATE[input_data], output_data)
            assert get_type(X) == output_data
        else:
            with pytest.raises(TypeError):
                X = convert_collection(
                    UNEQUAL_LENGTH_UNIVARIATE[input_data], output_data
                )


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
def test_convert_df_list(input_data):
    """Test that df list is correctly transposed."""
    X = convert_collection(EQUAL_LENGTH_UNIVARIATE[input_data], "df-list")
    assert X[0].shape == (20, 1)
    if input_data in EQUAL_LENGTH_MULTIVARIATE:
        X = convert_collection(EQUAL_LENGTH_MULTIVARIATE[input_data], "df-list")
        assert X[0].shape == (20, 2)


def test_resolve_equal_length_inner_type():
    test = ["numpy3D"]
    X = resolve_equal_length_inner_type(test)
    assert X == "numpy3D"
    test = ["np-list", "numpy3D", "FOOBAR"]
    X = resolve_equal_length_inner_type(test)
    assert X == "numpy3D"
    test = ["nested_univ", "np-list"]
    X = resolve_equal_length_inner_type(test)
    assert X == "np-list"


def test_resolve_unequal_length_inner_type():
    test = ["np-list"]
    X = resolve_unequal_length_inner_type(test)
    assert X == "np-list"
    test = ["np-list", "numpy3D"]
    X = resolve_unequal_length_inner_type(test)
    assert X == "np-list"
    test = ["nested_univ", "FOOBAR"]
    X = resolve_unequal_length_inner_type(test)
    assert X == "nested_univ"


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
    assert _equal_length(EQUAL_LENGTH_UNIVARIATE[data], data)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_equal_length(data):
    """Test if equal length series correctly identified."""
    assert is_equal_length(EQUAL_LENGTH_UNIVARIATE[data])


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not _equal_length(UNEQUAL_LENGTH_UNIVARIATE[data], data)


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_is_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not is_equal_length(UNEQUAL_LENGTH_UNIVARIATE[data])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_has_missing(data):
    assert not has_missing(EQUAL_LENGTH_UNIVARIATE[data])
    X = np.random.random(size=(10, 2, 20))
    X[5][1][12] = np.NAN
    # TODO: Test others with missing when converters complete
    assert has_missing(X)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_univariate(data):
    assert is_univariate(EQUAL_LENGTH_UNIVARIATE[data])
    if data in EQUAL_LENGTH_MULTIVARIATE.keys():
        assert not is_univariate(EQUAL_LENGTH_MULTIVARIATE[data])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test__is_nested_univ_dataframe(data):
    if data == "nested_univ":
        assert _is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not _is_nested_univ_dataframe(EQUAL_LENGTH_UNIVARIATE[data])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test__is_pd_wide(data):
    if data == "pd-wide":
        assert _is_pd_wide(EQUAL_LENGTH_UNIVARIATE[data])
    else:
        assert not _is_pd_wide(EQUAL_LENGTH_UNIVARIATE[data])
