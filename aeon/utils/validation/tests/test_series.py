"""Test series validation module."""

from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_series,
    make_example_pandas_series,
)
from aeon.utils.validation.series import (
    get_n_channels,
    get_n_timepoints,
    has_missing,
    is_series,
    is_univariate,
)


def test_is_series():
    """Test is_series validation."""
    assert not is_series(None)
    assert is_series(make_example_pandas_series())
    assert is_series(make_example_1d_numpy())
    assert is_series(make_example_dataframe_series(n_channels=1))
    assert is_series(make_example_dataframe_series(n_channels=2))
    assert is_series(make_example_2d_numpy_series())
    assert not is_series(make_example_3d_numpy())
    assert not is_series(make_example_3d_numpy_list())


def test_get_n_timepoints():
    """Test get_n_timepoints validation."""
    assert get_n_timepoints(make_example_pandas_series()) == 12
    assert get_n_timepoints(make_example_1d_numpy()) == 12
    assert get_n_timepoints(make_example_dataframe_series(n_channels=1), axis=1) == 12
    assert get_n_timepoints(make_example_dataframe_series(n_channels=2), axis=1) == 12
    assert get_n_timepoints(make_example_2d_numpy_series(n_channels=1), axis=1) == 12
    assert get_n_timepoints(make_example_2d_numpy_series(n_channels=2), axis=1) == 12


def test_get_n_channels():
    """Test get_n_channels validation."""
    assert get_n_channels(make_example_pandas_series()) == 1
    assert get_n_channels(make_example_1d_numpy()) == 1
    assert get_n_channels(make_example_dataframe_series(n_channels=1), axis=1) == 1
    assert get_n_channels(make_example_dataframe_series(n_channels=2), axis=1) == 2
    assert get_n_channels(make_example_2d_numpy_series(n_channels=1), axis=1) == 1
    assert get_n_channels(make_example_2d_numpy_series(n_channels=2), axis=1) == 2


def test_has_missing():
    """Test has_missing validation."""
    assert not has_missing(make_example_pandas_series())
    assert not has_missing(make_example_1d_numpy())
    assert not has_missing(make_example_dataframe_series(n_channels=1))
    assert not has_missing(make_example_dataframe_series(n_channels=2))
    assert not has_missing(make_example_2d_numpy_series(n_channels=1))
    assert not has_missing(make_example_2d_numpy_series(n_channels=2))


def test_is_univariate():
    """Test is_univariate validation."""
    assert is_univariate(make_example_pandas_series())
    assert is_univariate(make_example_1d_numpy())
    assert is_univariate(make_example_dataframe_series(n_channels=1), axis=1)
    assert not is_univariate(make_example_dataframe_series(n_channels=2), axis=1)
    assert not is_univariate(make_example_2d_numpy_series(n_channels=2), axis=1)
    assert not is_univariate(make_example_3d_numpy(), axis=1)
    assert not is_univariate(make_example_3d_numpy_list(), axis=1)
