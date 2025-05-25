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
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation.collection import (
    _is_numpy_list_multivariate,
    _is_pd_wide,
    get_n_cases,
    get_type,
    has_missing,
    is_equal_length,
    is_tabular,
    is_univariate,
)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_pd_wide(data):
    """Test _is_pd_wide function for different datatypes."""
    if data == "pd-wide":
        assert _is_pd_wide(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    else:
        assert not _is_pd_wide(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])


def test_is_tabular():
    """Test is_tabular function."""
    d1 = np.random.random(size=(10, 10))
    assert is_tabular(d1)
    assert is_tabular(pd.DataFrame(d1))
    assert not is_tabular(np.random.random(size=(10, 10, 10)))
    assert not is_tabular(np.random.random(size=(10)))


def test_get_type():
    """Test get_type function."""
    np_list = [np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8])]
    with pytest.raises(TypeError, match="np-list must contain 2D np.ndarray"):
        get_type(np_list)
    dp_list = [pd.DataFrame(np.random.random(size=(10, 10))), np.array([1, 2, 3, 4, 5])]
    with pytest.raises(TypeError, match="df-list must only contain pd.DataFrame"):
        get_type(dp_list)
    res = get_type(pd.DataFrame(np.random.random(size=(10, 10))))
    assert res == "pd-wide"
    data = {
        "Double_Column": [1.5, 2.3, 3.6, 4.8, 5.2],
        "String_Column": ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(TypeError, match="contain numeric values only"):
        get_type(df)


def test_has_missing():
    """Test has_missing function."""
    d1 = np.random.random(size=(10, 10))
    assert not has_missing(d1)
    d1[0, 0] = np.nan
    assert has_missing(d1)
    d2 = np.random.random(size=(10, 10))
    l1 = [d1, d1]
    assert has_missing(l1)
    l2 = [pd.DataFrame(d1), pd.DataFrame(d2)]
    assert has_missing(l2)


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

    # As the function is intended to be used for pairwise we assume it isnt a single
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


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_n_cases(data):
    """Test getting the number of cases."""
    assert get_n_cases(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == 10


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_type2(data):
    """Test getting the type."""
    assert get_type(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]) == data


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_equal_length(data):
    """Test if equal length series correctly identified."""
    assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])


@pytest.mark.parametrize("data", ["df-list", "np-list"])
def test_is_unequal_length(data):
    """Test if unequal length series correctly identified."""
    assert not is_equal_length(
        UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
    )


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_has_missing2(data):
    """Test if missing values are correctly identified."""
    assert not has_missing(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    X = np.random.random(size=(10, 2, 20))
    X[5][1][12] = np.nan
    assert has_missing(X)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_univariate(data):
    """Test if univariate series are correctly identified."""
    assert is_univariate(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])
    if data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION.keys():
        assert not is_univariate(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0]
        )
