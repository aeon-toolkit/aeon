"""Test check collection functionality."""

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
)
from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    MISSING_VALUES_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    UNIVARIATE_SERIES,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES, SERIES_DATA_TYPES
from aeon.utils.validation.collection import (
    _is_numpy_list_multivariate,
    get_n_cases,
    get_n_channels,
    get_n_timepoints,
    get_type,
    has_missing,
    is_collection,
    is_equal_length,
    is_tabular,
    is_univariate,
)


def test_is_tabular():
    """Test is_tabular function."""
    d1 = np.random.random(size=(10, 10))
    assert is_tabular(d1)
    assert is_tabular(pd.DataFrame(d1))
    assert not is_tabular(np.random.random(size=(10, 10, 10)))
    assert not is_tabular(np.random.random(size=(10)))


def test_is_collection():
    """Test is_collection function."""
    np_3d = np.random.random(size=(10, 10, 10))
    np_2d = np.random.random(size=(10, 10))
    assert is_collection(np_3d)
    assert not is_collection(np_2d)
    assert is_collection(np_2d, include_2d=True)
    assert not is_collection(None)


@pytest.mark.parametrize("data", SERIES_DATA_TYPES)
def test_is_collection_series(data):
    """Test is_collection function for series data types."""
    assert not is_collection(UNIVARIATE_SERIES[data]["train"][0])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_n_cases(data):
    """Test getting the number of cases."""
    assert get_n_cases(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == 10


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_n_timepoints(data):
    """Test getting the number of timepoints."""
    assert (
        get_n_timepoints(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == 20
    )
    if data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.keys():
        assert (
            get_n_timepoints(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
            is None
        )


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_n_channels(data):
    """Test getting the number of channels."""
    assert get_n_channels(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == 1
    if data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.keys():
        assert (
            get_n_channels(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0])
            == 2
        )


def test_get_n_channels_error():
    """Test error catching when getting the number of channels."""
    np_list = [
        np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
        np.array([[4, 5, 6, 7, 8], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
    ]
    with pytest.raises(ValueError, match="number of channels is not consistent"):
        get_n_channels(np_list)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_equal_length(data):
    """Test if equal length series correctly identified."""
    assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    if data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.keys():
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
        )


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_has_missing(data):
    """Test if missing values are correctly identified."""
    assert not has_missing(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    if data in MISSING_VALUES_CLASSIFICATION.keys():
        assert has_missing(MISSING_VALUES_CLASSIFICATION[data]["train"][0])


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_univariate(data):
    """Test if univariate series are correctly identified."""
    assert is_univariate(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    if data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.keys():
        assert not is_univariate(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0]
        )


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_type(data):
    """Test getting the collection data type."""
    assert get_type(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == data

    if data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.keys():
        assert (
            get_type(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == data
        )
    if data in MISSING_VALUES_CLASSIFICATION.keys():
        assert get_type(MISSING_VALUES_CLASSIFICATION[data]["train"][0]) == data
    if data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.keys():
        assert (
            get_type(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0]) == data
        )


def test_get_type_errors():
    """Test error catching in the get_type function."""
    with pytest.raises(TypeError, match="must be of type"):
        get_type(UNIVARIATE_SERIES["pd.Series"]["train"][0])

    assert (
        get_type(UNIVARIATE_SERIES["pd.Series"]["train"][0], raise_error=False) is None
    )

    np_list = [np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8])]
    with pytest.raises(TypeError, match="np-list must contain 2D np.ndarray"):
        get_type(np_list)

    dp_list = [pd.DataFrame(np.random.random(size=(10, 10))), np.array([1, 2, 3, 4, 5])]
    with pytest.raises(TypeError, match="df-list must only contain pd.DataFrame"):
        get_type(dp_list)

    data = {
        "Double_Column": [1.5, 2.3, 3.6, 4.8, 5.2],
        "String_Column": ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(TypeError, match="contain numeric values only"):
        get_type(df)


def test_is_numpy_list_multivariate_single():
    """Test collection of multivariate numpy list."""
    # 3d format tests
    # Equal (n_cases, n_channels, n_timepoints)
    x_univ = make_example_3d_numpy(10, 1, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ)
    assert is_multivariate is False

    # Unequal List[(n_channels, n_timepoints)]
    x_univ_unequal = make_example_3d_numpy_list(10, 1, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ_unequal)
    assert is_multivariate is False

    # Equal numba list NumbaList[(n_channels, n_timepoints)]
    x_univ_numba_list = NumbaList(x_univ)
    is_multivariate = _is_numpy_list_multivariate(x_univ_numba_list)
    assert is_multivariate is False

    # Unequal numba list NumbaList[(n_channels, n_timepoints)]
    x_univ_unequal_numba_list = NumbaList(x_univ_unequal)
    is_multivariate = _is_numpy_list_multivariate(x_univ_unequal_numba_list)
    assert is_multivariate is False

    # 2d format tests
    # Equal (n_cases, n_timepoints)
    x_univ_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ_2d)
    assert is_multivariate is False

    # Unequal List[(n_timepoints)]
    x_univ_2d_unequal = make_example_2d_numpy_list(10, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ_2d_unequal)
    assert is_multivariate is False

    # Equal numba list NumbaList[(n_timepoints)]
    x_univ_2d_numba_list = NumbaList(x_univ_2d)
    is_multivariate = _is_numpy_list_multivariate(x_univ_2d_numba_list)
    assert is_multivariate is False

    # Unequal numba list NumbaList[(n_timepoints)]
    x_univ_2d_unequal_numba_list = NumbaList(x_univ_2d_unequal)
    is_multivariate = _is_numpy_list_multivariate(x_univ_2d_unequal_numba_list)
    assert is_multivariate is False

    # 1d format tests
    # Equal (n_timepoints)
    x_univ_1d = make_example_1d_numpy(10)
    is_multivariate = _is_numpy_list_multivariate(x_univ_1d)
    assert is_multivariate is False

    # Equal NumbaList[n_timepoints]
    x_univ_numba_list_1d = NumbaList(x_univ_1d)
    is_multivariate = _is_numpy_list_multivariate(x_univ_numba_list_1d)
    assert is_multivariate is False

    # 3d format tests multivariate
    # Equal (n_cases, n_channels, n_timepoints)
    x_multi = make_example_3d_numpy(10, 5, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_multi)
    assert is_multivariate is True

    # Unequal List[(n_channels, n_timepoints)]
    x_multi_unequal = make_example_3d_numpy_list(10, 5, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_multi_unequal)
    assert is_multivariate is True

    # Equal numba list NumbaList[(n_channels, n_timepoints)]
    x_multi_numba_list = NumbaList(x_multi)
    is_multivariate = _is_numpy_list_multivariate(x_multi_numba_list)
    assert is_multivariate is True

    # Unequal numba list NumbaList[(n_channels, n_timepoints)]
    x_multi_unequal_numba_list = NumbaList(x_multi_unequal)
    is_multivariate = _is_numpy_list_multivariate(x_multi_unequal_numba_list)
    assert is_multivariate is True

    # 2d format tests
    # Equal (n_cases, n_timepoints)

    # As the function is intended to be used for pairwise we assume it isn't a single
    # multivariate time series but two collections of univariate
    x_multi_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_multi_2d)
    assert is_multivariate is False

    x_multi_2d_numba_list = NumbaList(x_multi_2d)
    is_multivariate = _is_numpy_list_multivariate(x_multi_2d_numba_list)
    assert is_multivariate is False

    with pytest.raises(ValueError, match="The format of you input is not supported."):
        _is_numpy_list_multivariate(1.0)

    with pytest.raises(ValueError, match="The format of you input is not supported."):
        _is_numpy_list_multivariate(1.0, x_multi_2d)

    with pytest.raises(ValueError, match="The format of you input is not supported."):
        _is_numpy_list_multivariate(x_multi_2d, 1.0)


def test_is_numpy_list_multivariate_two_univ():
    """Test collection of two univariate numpy list."""
    # 3d format tests
    # Equal (n_cases, n_channels, n_timepoints)
    x_univ = make_example_3d_numpy(10, 1, 20, return_y=False)
    y_univ = make_example_3d_numpy(10, 1, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ, y_univ)
    assert is_multivariate is False

    # Unequal List[(n_channels, n_timepoints)]
    x_univ_unequal = make_example_3d_numpy_list(10, 1, return_y=False)
    y_univ_unequal = make_example_3d_numpy_list(10, 1, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ_unequal, y_univ_unequal)
    assert is_multivariate is False

    # Equal numba list NumbaList[(n_channels, n_timepoints)]
    x_univ_numba_list = NumbaList(x_univ)
    y_univ_numba_list = NumbaList(y_univ)
    is_multivariate = _is_numpy_list_multivariate(x_univ_numba_list, y_univ_numba_list)
    assert is_multivariate is False

    # Unequal numba list NumbaList[(n_channels, n_timepoints)]
    x_univ_unequal_numba_list = NumbaList(x_univ_unequal)
    y_univ_unequal_numba_list = NumbaList(y_univ_unequal)
    is_multivariate = _is_numpy_list_multivariate(
        x_univ_unequal_numba_list, y_univ_unequal_numba_list
    )
    assert is_multivariate is False

    # 2d format tests
    # Equal (n_cases, n_timepoints)
    x_univ_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    y_univ_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ_2d, y_univ_2d)
    assert is_multivariate is False

    # Unequal List[(n_timepoints)]
    x_univ_2d_unequal = make_example_2d_numpy_list(10, return_y=False)
    y_univ_2d_unequal = make_example_2d_numpy_list(10, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_univ_2d_unequal, y_univ_2d_unequal)
    assert is_multivariate is False

    # Equal numba list NumbaList[(n_timepoints)]
    x_univ_2d_numba_list = NumbaList(x_univ_2d)
    y_univ_2d_numba_list = NumbaList(y_univ_2d)
    is_multivariate = _is_numpy_list_multivariate(
        x_univ_2d_numba_list, y_univ_2d_numba_list
    )
    assert is_multivariate is False

    # Unequal numba list NumbaList[(n_timepoints)]
    x_univ_2d_unequal_numba_list = NumbaList(x_univ_2d_unequal)
    y_univ_2d_unequal_numba_list = NumbaList(y_univ_2d_unequal)
    is_multivariate = _is_numpy_list_multivariate(
        x_univ_2d_unequal_numba_list, y_univ_2d_unequal_numba_list
    )
    assert is_multivariate is False

    # 1d format tests
    # Equal (n_timepoints)
    x_univ_1d = make_example_1d_numpy(10)
    y_univ_1d = make_example_1d_numpy(10)
    is_multivariate = _is_numpy_list_multivariate(x_univ_1d, y_univ_1d)
    assert is_multivariate is False

    # Equal NumbaList[n_timepoints]
    x_univ_numba_list_1d = NumbaList(x_univ_1d)
    y_univ_numba_list_1d = NumbaList(y_univ_1d)
    is_multivariate = _is_numpy_list_multivariate(
        x_univ_numba_list_1d, y_univ_numba_list_1d
    )
    assert is_multivariate is False

    # Test single to multiple
    is_multivariate = _is_numpy_list_multivariate(x_univ, x_univ_2d)
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(x_univ_2d, x_univ)
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(
        x_univ_numba_list, x_univ_2d_numba_list
    )
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(
        x_univ_2d_numba_list, x_univ_numba_list
    )
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(x_univ_2d, x_univ_1d)
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(x_univ, x_univ_1d)
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(x_univ_1d, x_univ_2d)
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(
        x_univ_2d_numba_list, x_univ_numba_list_1d
    )
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(
        x_univ_numba_list_1d, x_univ_2d_numba_list
    )
    assert is_multivariate is False

    is_multivariate = _is_numpy_list_multivariate(
        x_univ_numba_list, x_univ_numba_list_1d
    )
    assert is_multivariate is False


def test_is_numpy_list_multivariate_two_multi():
    """Test collection of two multivariate numpy list."""
    # 3d format tests multivariate
    # Equal (n_cases, n_channels, n_timepoints)
    x_multi = make_example_3d_numpy(10, 5, 20, return_y=False)
    y_multi = make_example_3d_numpy(10, 5, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_multi, y_multi)
    assert is_multivariate is True

    # Unequal List[(n_channels, n_timepoints)]
    x_multi_unequal = make_example_3d_numpy_list(10, 5, return_y=False)
    y_multi_unequal = make_example_3d_numpy_list(10, 5, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_multi_unequal, y_multi_unequal)
    assert is_multivariate is True

    # Equal numba list NumbaList[(n_channels, n_timepoints)]
    x_multi_numba_list = NumbaList(x_multi)
    y_multi_numba_list = NumbaList(y_multi)
    is_multivariate = _is_numpy_list_multivariate(
        x_multi_numba_list, y_multi_numba_list
    )
    assert is_multivariate is True

    # Unequal numba list NumbaList[(n_channels, n_timepoints)]
    x_multi_unequal_numba_list = NumbaList(x_multi_unequal)
    y_multi_unequal_numba_list = NumbaList(y_multi_unequal)
    is_multivariate = _is_numpy_list_multivariate(
        x_multi_unequal_numba_list, y_multi_unequal_numba_list
    )
    assert is_multivariate is True

    x_multi_2d = make_example_2d_numpy_collection(10, 20, return_y=False)
    is_multivariate = _is_numpy_list_multivariate(x_multi, x_multi_2d)
    assert is_multivariate is True

    x_multi_2d_numba_list = NumbaList(x_multi_2d)
    is_multivariate = _is_numpy_list_multivariate(
        x_multi_numba_list, x_multi_2d_numba_list
    )
    assert is_multivariate is True
