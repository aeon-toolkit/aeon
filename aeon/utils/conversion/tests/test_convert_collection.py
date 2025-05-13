"""Unit tests for check/convert functions."""

from copy import deepcopy

import numpy as np
import pytest

from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
)
from aeon.testing.utils.deep_equals import deep_equals
from aeon.utils.conversion._convert_collection import (
    _from_numpy2d_to_df_list,
    _from_numpy2d_to_np_list,
    _from_numpy2d_to_numpy3d,
    _from_numpy2d_to_pd_multiindex,
    _from_numpy2d_to_pd_wide,
    _from_numpy3d_to_df_list,
    _from_numpy3d_to_np_list,
    _from_numpy3d_to_numpy2d,
    _from_numpy3d_to_pd_multiindex,
    _from_numpy3d_to_pd_wide,
    convert_collection,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)
from aeon.utils.data_types import (
    COLLECTIONS_DATA_TYPES,
    COLLECTIONS_MULTIVARIATE_DATA_TYPES,
    COLLECTIONS_UNEQUAL_DATA_TYPES,
)
from aeon.utils.validation import get_type


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
@pytest.mark.parametrize("output_data", COLLECTIONS_DATA_TYPES)
def test_convert_collection(input_data, output_data):
    """Test all valid and invalid conversions."""
    # All should work with univariate equal length
    X = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0]
    Xc = convert_collection(X, output_data)
    assert get_type(Xc) == output_data
    assert _conversion_shape_3d(X, input_data) == _conversion_shape_3d(Xc, output_data)

    # Test with multivariate
    if input_data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        if output_data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
            X = EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0]
            Xc = convert_collection(X, output_data)
            assert get_type(Xc) == output_data
            assert _conversion_shape_3d(X, input_data) == _conversion_shape_3d(
                Xc, output_data
            )
        else:
            with pytest.raises(TypeError, match="Cannot convert multivariate"):
                convert_collection(
                    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0],
                    output_data,
                )

    # Test with unequal length
    if input_data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        if output_data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
            X = UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0]
            Xc = convert_collection(
                X,
                output_data,
            )
            assert get_type(Xc) == output_data
            assert _conversion_shape_3d(X, input_data) == _conversion_shape_3d(
                Xc, output_data
            )
        else:
            with pytest.raises(TypeError, match="Cannot convert unequal"):
                convert_collection(
                    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0],
                    output_data,
                )


def _conversion_shape_3d(X, input_data):
    if input_data == "numpy3D":
        return X.shape
    elif input_data == "numpy2D" or input_data == "pd-wide":
        return X.shape[0], 1, X.shape[1]
    elif input_data == "pd-multiindex":
        return (
            len(X.index.get_level_values(0).unique()),
            X.columns.nunique(),
            X.loc[X.index.get_level_values(0).unique()[-1]].index.nunique(),
        )
    elif input_data == "df-list" or input_data == "np-list":
        return len(X), X[-1].shape[0], X[-1].shape[1]
    else:
        raise TypeError(f"Unknown data type: {input_data}")


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
def test_self_conversion(input_data):
    """Test that data is correctly copied when converting to same data type."""
    X = deepcopy(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0])
    Xc = convert_collection(
        EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0], input_data
    )
    assert X is not Xc
    assert deep_equals(X, Xc)


@pytest.mark.parametrize("input_data", COLLECTIONS_DATA_TYPES)
def test_conversion_loop_returns_same_data(input_data):
    """Test that chaining conversions ending at the start gives the same data."""
    dtypes = COLLECTIONS_DATA_TYPES.copy()
    np.random.shuffle(dtypes)
    Xc = deepcopy(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0])
    for i in dtypes:
        Xc = convert_collection(Xc, i)
    Xc = convert_collection(Xc, input_data)

    eq, msg = deep_equals(
        EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0],
        Xc,
        ignore_index=True,
        return_msg=True,
    )
    assert eq, msg


@pytest.mark.parametrize("input_data", COLLECTIONS_MULTIVARIATE_DATA_TYPES)
def test_conversion_loop_returns_same_data_multivariate(input_data):
    """Test that chaining conversions ending at the start gives the same data."""
    dtypes = COLLECTIONS_MULTIVARIATE_DATA_TYPES.copy()
    np.random.shuffle(dtypes)
    Xc = deepcopy(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0])
    for i in dtypes:
        Xc = convert_collection(Xc, i)
    Xc = convert_collection(Xc, input_data)

    eq, msg = deep_equals(
        EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[input_data]["train"][0],
        Xc,
        ignore_index=True,
        return_msg=True,
    )
    assert eq, msg


@pytest.mark.parametrize("input_data", COLLECTIONS_UNEQUAL_DATA_TYPES)
def test_conversion_loop_returns_same_data_unequal(input_data):
    """Test that chaining conversions ending at the start gives the same data."""
    dtypes = COLLECTIONS_UNEQUAL_DATA_TYPES.copy()
    np.random.shuffle(dtypes)
    Xc = deepcopy(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0])
    for i in dtypes:
        Xc = convert_collection(Xc, i)
    Xc = convert_collection(Xc, input_data)

    eq, msg = deep_equals(
        UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[input_data]["train"][0],
        Xc,
        ignore_index=True,
        return_msg=True,
    )
    assert eq, msg


def test_resolve_equal_length_inner_type():
    """Test the resolution of inner type for equal length collections."""
    for input in COLLECTIONS_DATA_TYPES:
        X = resolve_equal_length_inner_type([input])
        assert X == input

    test = ["numpy3D"]
    X = resolve_equal_length_inner_type(test)
    assert X == "numpy3D"
    test = ["np-list", "numpy3D", "FOOBAR"]
    X = resolve_equal_length_inner_type(test)
    assert X == "numpy3D"
    test = ["pd-wide", "np-list"]
    X = resolve_equal_length_inner_type(test)
    assert X == "np-list"

    with pytest.raises(ValueError, match="no valid inner types"):
        resolve_equal_length_inner_type(["invalid"])


def test_resolve_unequal_length_inner_type():
    """Test the resolution of inner type for unequal length collections."""
    for input in COLLECTIONS_UNEQUAL_DATA_TYPES:
        X = resolve_unequal_length_inner_type([input])
        assert X == input

    test = ["np-list"]
    X = resolve_unequal_length_inner_type(test)
    assert X == "np-list"
    test = ["np-list", "numpy3D"]
    X = resolve_unequal_length_inner_type(test)
    assert X == "np-list"

    with pytest.raises(ValueError, match="no valid inner types"):
        resolve_unequal_length_inner_type(["numpy3D"])


NUMPY3D = [
    _from_numpy3d_to_pd_wide,
    _from_numpy3d_to_np_list,
    _from_numpy3d_to_df_list,
    _from_numpy3d_to_pd_wide,
    _from_numpy3d_to_numpy2d,
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
    _from_numpy2d_to_pd_multiindex,
]


@pytest.mark.parametrize("function", NUMPY2D)
def test_numpy2D_error(function):
    """Test numpy flat converters only work with 2D numpy."""
    X = np.random.random(size=(10, 2, 20))
    with pytest.raises(TypeError, match="Input numpy not of type numpy2D"):
        function(X)
