"""Test series validation module."""

import numpy as np
import pandas as pd
import pytest

from aeon.testing.testing_data import (
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    MISSING_VALUES_SERIES,
    MULTIVARIATE_SERIES,
    UNIVARIATE_SERIES,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES, SERIES_DATA_TYPES
from aeon.utils.validation.series import (
    get_n_channels,
    get_n_timepoints,
    get_type,
    has_missing,
    is_series,
    is_univariate,
)


def test_is_series():
    """Test is_series function."""
    np_1d = np.random.random(size=(10))
    series = pd.Series(np_1d)
    np_2d = np.random.random(size=(10, 10))
    assert is_series(np_1d)
    assert is_series(series)
    assert not is_series(np_2d)
    assert is_series(np_2d, include_2d=True)
    assert not is_series(None)


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_is_series_collection(data):
    """Test is_series function for collection data types."""
    assert not is_series(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0])


@pytest.mark.parametrize("data", SERIES_DATA_TYPES)
def test_get_n_timepoints(data):
    """Test getting the number of timepoints."""
    assert get_n_timepoints(UNIVARIATE_SERIES[data]["train"][0], axis=1) == 20
    if data in MULTIVARIATE_SERIES.keys():
        assert get_n_timepoints(MULTIVARIATE_SERIES[data]["train"][0], axis=1) == 20


@pytest.mark.parametrize("data", SERIES_DATA_TYPES)
def test_get_n_channels(data):
    """Test getting the number of channels."""
    assert get_n_channels(UNIVARIATE_SERIES[data]["train"][0], axis=1) == 1
    if data in MULTIVARIATE_SERIES.keys():
        assert get_n_channels(MULTIVARIATE_SERIES[data]["train"][0], axis=1) == 2


@pytest.mark.parametrize("data", SERIES_DATA_TYPES)
def test_has_missing(data):
    """Test if missing values are correctly identified."""
    assert not has_missing(UNIVARIATE_SERIES[data]["train"][0])
    if data in MISSING_VALUES_SERIES.keys():
        assert has_missing(MISSING_VALUES_SERIES[data]["train"][0])


@pytest.mark.parametrize("data", SERIES_DATA_TYPES)
def test_is_univariate(data):
    """Test if univariate series are correctly identified."""
    assert is_univariate(UNIVARIATE_SERIES[data]["train"][0], axis=1)
    if data in MULTIVARIATE_SERIES.keys():
        assert not is_univariate(MULTIVARIATE_SERIES[data]["train"][0], axis=1)


@pytest.mark.parametrize("data", SERIES_DATA_TYPES)
def test_get_type(data):
    """Test getting the collection data type."""
    assert get_type(UNIVARIATE_SERIES[data]["train"][0]) == data

    if data in MISSING_VALUES_SERIES.keys():
        assert get_type(MISSING_VALUES_SERIES[data]["train"][0]) == data
    if data in MULTIVARIATE_SERIES.keys():
        assert get_type(MULTIVARIATE_SERIES[data]["train"][0]) == data


def test_get_type_errors():
    """Test error catching in the get_type function."""
    with pytest.raises(TypeError, match="must be of type"):
        get_type(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["np-list"]["train"][0])

    assert (
        get_type(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["np-list"]["train"][0],
            raise_error=False,
        )
        is None
    )

    data = {
        "Double_Column": [1.5, 2.3, 3.6, 4.8, 5.2],
        "String_Column": ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(TypeError, match="contain numeric values only"):
        get_type(df)
