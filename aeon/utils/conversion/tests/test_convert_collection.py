"""Unit tests for check/convert functions."""

import numpy as np
import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_example_nested_dataframe
from aeon.testing.utils.data_gen._collection import (
    EQUAL_LENGTH_MULTIVARIATE,
    EQUAL_LENGTH_UNIVARIATE,
    UNEQUAL_LENGTH_UNIVARIATE,
)
from aeon.utils import COLLECTIONS_DATA_TYPES
from aeon.utils.conversion._convert_collection import (
    _from_nested_univ_to_numpy2d,
    _from_nested_univ_to_pd_multiindex,
    _from_numpy2d_to_df_list,
    _from_numpy2d_to_nested_univ,
    _from_numpy2d_to_np_list,
    _from_numpy2d_to_numpy3d,
    _from_numpy2d_to_pd_multiindex,
    _from_numpy2d_to_pd_wide,
    _from_numpy3d_to_df_list,
    _from_numpy3d_to_nested_univ,
    _from_numpy3d_to_np_list,
    _from_numpy3d_to_numpy2d,
    _from_numpy3d_to_pd_multiindex,
    _from_numpy3d_to_pd_wide,
    convert_collection,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)
from aeon.utils.validation.collection import (
    _equal_length,
    _nested_univ_is_equal,
    get_n_cases,
    get_type,
    has_missing,
    is_equal_length,
    is_univariate,
)


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
@pytest.mark.parametrize("output_data", COLLECTIONS_DATA_TYPES)
def test_convert_collection(input_data, output_data):
    """Test all valid and invalid conversions."""
    # All should work with univariate equal length
    X = convert_collection(EQUAL_LENGTH_UNIVARIATE[input_data], output_data)
    assert get_type(X) == output_data
    # Test with multivariate
    if input_data in EQUAL_LENGTH_MULTIVARIATE:
        if output_data in EQUAL_LENGTH_MULTIVARIATE:
            X = convert_collection(EQUAL_LENGTH_MULTIVARIATE[input_data], output_data)
            assert get_type(X) == output_data
        else:
            with pytest.raises(TypeError, match="Cannot convert multivariate"):
                X = convert_collection(
                    EQUAL_LENGTH_MULTIVARIATE[input_data], output_data
                )
    # Test with unequal length
    if input_data in UNEQUAL_LENGTH_UNIVARIATE:
        if output_data in UNEQUAL_LENGTH_UNIVARIATE or output_data == "pd-multiindex":
            X = convert_collection(UNEQUAL_LENGTH_UNIVARIATE[input_data], output_data)
            assert get_type(X) == output_data
        else:
            with pytest.raises(TypeError, match="Cannot convert unequal"):
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
    """Test the resolution of inner type for equal length collections."""
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
    """Test the resolution of inner type for unequal length collections."""
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
    """Test if missing values are correctly identified."""
    assert not has_missing(EQUAL_LENGTH_UNIVARIATE[data])
    X = np.random.random(size=(10, 2, 20))
    X[5][1][12] = np.NAN
    assert has_missing(X)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_univariate(data):
    """Test if univariate series are correctly identified."""
    assert is_univariate(EQUAL_LENGTH_UNIVARIATE[data])
    if data in EQUAL_LENGTH_MULTIVARIATE.keys():
        assert not is_univariate(EQUAL_LENGTH_MULTIVARIATE[data])


NUMPY3D = [
    _from_numpy3d_to_pd_wide,
    _from_numpy3d_to_np_list,
    _from_numpy3d_to_df_list,
    _from_numpy3d_to_pd_wide,
    _from_numpy3d_to_numpy2d,
    _from_numpy3d_to_nested_univ,
    _from_numpy3d_to_pd_multiindex,
]


@pytest.mark.parametrize("function", NUMPY3D)
def test_numpy3D_error(function):
    """Test input type error for numpy3D."""
    X = np.random.random(size=(10, 20))
    with pytest.raises(TypeError, match="Input should be 3-dimensional NumPy array"):
        function(X)


NUMPY2D = [
    _from_numpy2d_to_numpy3d,
    _from_numpy2d_to_np_list,
    _from_numpy2d_to_df_list,
    _from_numpy2d_to_pd_wide,
    _from_numpy2d_to_nested_univ,
    _from_numpy2d_to_pd_multiindex,
]


@pytest.mark.parametrize("function", NUMPY2D)
def test_numpy2D_error(function):
    """Test numpy flat converters only work with 2D numpy."""
    X = np.random.random(size=(10, 2, 20))
    with pytest.raises(TypeError, match="Input numpy not of type numpy2D"):
        function(X)


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


def test_from_nested():
    """Test with multiple nested columns and non-nested columns."""
    data = {
        "A": [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])],
        "B": [pd.Series(["a", "b", "c"]), pd.Series(["x", "y", "z"])],
        "C": [7, 8],
    }
    X = pd.DataFrame(data)
    result = _from_nested_univ_to_pd_multiindex(X)
    assert isinstance(result, pd.DataFrame)
    data = {
        "A": [pd.Series([1, 2, 3, 4]), pd.Series([4, 5, 6])],
        "B": [pd.Series([1, 2, 3, 4]), pd.Series([4, 5, 6])],
        "C": [pd.Series([1, 2, 3, 4]), pd.Series([4, 5, 6])],
    }
    X = pd.DataFrame(data)
    with pytest.raises(
        TypeError, match="Cannot convert unequal length series to numpy2D"
    ):
        _from_nested_univ_to_numpy2d(X)
    X, _ = make_example_nested_dataframe(n_cases=10, n_channels=1, n_timepoints=20)
    result = _from_nested_univ_to_numpy2d(X)
    assert result.shape == (10, 20)
    X, _ = make_example_nested_dataframe(n_cases=10, n_channels=2, n_timepoints=20)
    with pytest.raises(
        TypeError, match="Cannot convert multivariate nested into numpy2D"
    ):
        _from_nested_univ_to_numpy2d(X)
