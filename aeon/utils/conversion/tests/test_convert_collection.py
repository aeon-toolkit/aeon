"""Unit tests for check/convert functions."""

import numpy as np
import pandas as pd
import pytest
from numba.typed import List as NumbaList

from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_list,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_nested_dataframe,
)
from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
)
from aeon.utils import COLLECTIONS_DATA_TYPES
from aeon.utils.conversion._convert_collection import (
    _convert_collection_to_numba_list,
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
    X = convert_collection(
        EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0], output_data
    )
    assert get_type(X) == output_data
    # Test with multivariate
    if input_data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        if output_data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
            X = convert_collection(
                EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0],
                output_data,
            )
            assert get_type(X) == output_data
        else:
            with pytest.raises(TypeError, match="Cannot convert multivariate"):
                X = convert_collection(
                    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0],
                    output_data,
                )
    # Test with unequal length
    if input_data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        if (
            output_data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
            or output_data == "pd-multiindex"
        ):
            X = convert_collection(
                UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0],
                output_data,
            )
            assert get_type(X) == output_data
        else:
            with pytest.raises(TypeError, match="Cannot convert unequal"):
                X = convert_collection(
                    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0],
                    output_data,
                )


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
def test_convert_df_list(input_data):
    """Test that df list is correctly transposed."""
    X = convert_collection(
        EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0], "df-list"
    )
    assert X[0].shape == (20, 1)
    if input_data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        X = convert_collection(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0], "df-list"
        )
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
    assert get_n_cases(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == 10


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_type(data):
    """Test getting the type."""
    assert get_type(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == data


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_equal_length(data):
    """Test if equal length series correctly identified."""
    assert _equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0], data)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_equal_length(data):
    """Test if equal length series correctly identified."""
    assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not _equal_length(
        UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0], data
    )


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_is_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not is_equal_length(
        UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
    )


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_has_missing(data):
    """Test if missing values are correctly identified."""
    assert not has_missing(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    X = np.random.random(size=(10, 2, 20))
    X[5][1][12] = np.NAN
    assert has_missing(X)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_univariate(data):
    """Test if univariate series are correctly identified."""
    assert is_univariate(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    if data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.keys():
        assert not is_univariate(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0]
        )


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
    X, _ = make_example_nested_dataframe(
        n_cases=10, n_channels=1, min_n_timepoints=20, max_n_timepoints=20
    )
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
    X, _ = make_example_nested_dataframe(
        n_cases=10, n_channels=1, min_n_timepoints=20, max_n_timepoints=20
    )
    result = _from_nested_univ_to_numpy2d(X)
    assert result.shape == (10, 20)
    X, _ = make_example_nested_dataframe(
        n_cases=10, n_channels=2, min_n_timepoints=20, max_n_timepoints=20
    )
    with pytest.raises(
        TypeError, match="Cannot convert multivariate nested into numpy2D"
    ):
        _from_nested_univ_to_numpy2d(X)


def test_collection_to_numba_list_univariate():
    """Test collection of univariate numba list."""
    # 3d format tests
    # Equal (n_cases, n_channels, n_timepoints)
    x_univ = make_example_3d_numpy(10, 1, 20, return_y=False)
    convert_x_univ, unequal = _convert_collection_to_numba_list(x_univ)
    assert isinstance(convert_x_univ, NumbaList)
    assert unequal is False

    # Unequal List[(n_channels, n_timepoints)]
    x_univ_unequal = make_example_3d_numpy_list(10, 1, return_y=False)
    convert_x_unequal_univ, unequal = _convert_collection_to_numba_list(x_univ_unequal)
    assert isinstance(convert_x_unequal_univ, NumbaList)
    assert unequal is True

    # Equal numba list NumbaList[(n_channels, n_timepoints)]
    x_univ_numba_list = NumbaList(x_univ)
    convert_x_univ_numba_list, unequal = _convert_collection_to_numba_list(
        x_univ_numba_list
    )
    assert isinstance(convert_x_univ_numba_list, NumbaList)
    assert unequal is False

    # Unequal numba list NumbaList[(n_channels, n_timepoints)]
    x_univ_unequal_numba_list = NumbaList(x_univ_unequal)
    convert_x_unequal_univ_numba_list, unequal = _convert_collection_to_numba_list(
        x_univ_unequal_numba_list
    )
    assert isinstance(convert_x_unequal_univ_numba_list, NumbaList)
    assert unequal is True

    # 2d format tests
    # Equal (n_cases, n_timepoints)
    x_univ_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    convert_x_univ_2d, unequal = _convert_collection_to_numba_list(x_univ_2d)
    assert isinstance(convert_x_univ_2d, NumbaList)
    assert unequal is False

    # Unequal List[(n_timepoints)]
    x_univ_2d_unequal = make_example_2d_numpy_list(10, return_y=False)
    convert_x_unequal_univ_2d, unequal = _convert_collection_to_numba_list(
        x_univ_2d_unequal
    )
    assert isinstance(convert_x_unequal_univ_2d, NumbaList)
    assert unequal is True

    # Equal numba list NumbaList[(n_timepoints)]
    x_univ_2d_numba_list = NumbaList(x_univ_2d)
    convert_x_univ_2d_numba_list, unequal = _convert_collection_to_numba_list(
        x_univ_2d_numba_list
    )
    assert isinstance(convert_x_univ_2d_numba_list, NumbaList)
    assert unequal is False

    # Unequal numba list NumbaList[(n_timepoints)]
    x_univ_2d_unequal_numba_list = NumbaList(x_univ_2d_unequal)
    convert_x_unequal_univ_2d_numba_list, unequal = _convert_collection_to_numba_list(
        x_univ_2d_unequal_numba_list
    )
    assert isinstance(convert_x_unequal_univ_2d_numba_list, NumbaList)
    assert unequal is True

    # 1d format tests
    # Equal (n_timepoints)
    x_univ_1d = make_example_1d_numpy(10)
    convert_x_univ_1d, unequal = _convert_collection_to_numba_list(x_univ_1d)
    assert isinstance(convert_x_univ_1d, NumbaList)
    assert unequal is False

    # Equal NumbaList[n_timepoints]
    x_univ_numba_list_1d = NumbaList(x_univ_1d)
    convert_x_univ_numba_list_1d, unequal = _convert_collection_to_numba_list(
        x_univ_numba_list_1d
    )
    assert isinstance(convert_x_univ_numba_list_1d, NumbaList)
    assert unequal is False


def test_collection_to_numba_list_multivariate():
    """Test collection of multivariate numba list."""
    # 3d format tests
    # Equal (n_cases, n_channels, n_timepoints)
    x_multi = make_example_3d_numpy(10, 5, 20, return_y=False)
    convert_x_multi, unequal = _convert_collection_to_numba_list(
        x_multi, multivariate_conversion=True
    )
    assert isinstance(convert_x_multi, NumbaList)
    assert unequal is False

    # Unequal List[(n_channels, n_timepoints)]
    x_multi_unequal = make_example_3d_numpy_list(10, 5, return_y=False)
    convert_x_unequal_multi, unequal = _convert_collection_to_numba_list(
        x_multi_unequal, multivariate_conversion=True
    )
    assert isinstance(convert_x_unequal_multi, NumbaList)
    assert unequal is True

    # Equal numba list NumbaList[(n_channels, n_timepoints)]
    x_multi_numba_list = NumbaList(x_multi)
    convert_x_multi_numba_list, unequal = _convert_collection_to_numba_list(
        x_multi_numba_list
    )
    assert isinstance(convert_x_multi_numba_list, NumbaList)
    assert unequal is False

    # Unequal numba list NumbaList[(n_channels, n_timepoints)]
    x_multi_unequal_numba_list = NumbaList(x_multi_unequal)
    convert_x_unequal_multi_numba_list, unequal = _convert_collection_to_numba_list(
        x_multi_unequal_numba_list
    )
    assert isinstance(convert_x_unequal_multi_numba_list, NumbaList)
    assert unequal is True

    # 2d format tests
    # Equal (n_channels, n_timepoints)
    x_multi_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    convert_x_multi_2d, unequal = _convert_collection_to_numba_list(
        x_multi_2d, multivariate_conversion=True
    )
    assert isinstance(convert_x_multi_2d, NumbaList)
    assert unequal is False

    # NumbaList(n_channels,
    x_multi_2d_numba_list = NumbaList(x_multi_2d)
    convert_x_multi_2d_numba_list, unequal = _convert_collection_to_numba_list(
        x_multi_2d_numba_list
    )
    assert isinstance(convert_x_multi_2d_numba_list, NumbaList)
    assert unequal is False

    with pytest.raises(ValueError, match="example must be 1D, 2D or 3D"):
        # make 4d array
        x_multi_4d = np.zeros((10, 5, 20, 3))
        _convert_collection_to_numba_list(
            x_multi_4d, multivariate_conversion=True, name="example"
        )

    with pytest.raises(ValueError, match="example must include only 1D or 2D arrays"):
        x_multi_4d_list = list(np.zeros((10, 5, 20, 3)))
        _convert_collection_to_numba_list(
            x_multi_4d_list, multivariate_conversion=True, name="example"
        )

    with pytest.raises(
        ValueError,
        match="example must be either np.ndarray or "
        r"List\[np.ndarray\] or "
        r"NumbaList\[np.ndarray\]",
    ):
        _convert_collection_to_numba_list(
            10, multivariate_conversion=True, name="example"
        )
