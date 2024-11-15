"""Test series module."""

__maintainer__ = ["TonyBagnall"]

import numpy as np
import pandas as pd
import pytest

from aeon.testing.data_generation import (
    _make_hierarchical,
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_series,
    make_example_pandas_series,
)
from aeon.utils.validation.series import (
    check_series,
    is_hierarchical,
    is_single_series,
    is_univariate_series,
)


def test_is_univariate_series():
    """Test is_univariate_series."""
    assert not is_univariate_series(None)
    assert is_univariate_series(make_example_pandas_series())
    assert is_univariate_series(make_example_1d_numpy())
    assert is_univariate_series(make_example_dataframe_series(n_channels=1))
    assert not is_univariate_series(make_example_dataframe_series(n_channels=2))
    assert not is_univariate_series(make_example_2d_numpy_series())
    assert not is_univariate_series(make_example_3d_numpy())
    assert not is_univariate_series(make_example_3d_numpy_list())


def test_is_hierarchical():
    """Test is_hierarchical."""
    assert is_hierarchical(_make_hierarchical())
    assert not is_hierarchical(make_example_dataframe_series(n_channels=2))
    assert not is_hierarchical(make_example_1d_numpy())
    assert not is_hierarchical(make_example_2d_numpy_series())
    assert not is_hierarchical(make_example_3d_numpy_list())


def test_is_single_series():
    """Test is_single_series."""
    assert not is_single_series(None)
    assert is_single_series(make_example_pandas_series())
    assert is_single_series(make_example_1d_numpy())
    assert is_univariate_series(make_example_dataframe_series(n_channels=1))
    assert is_single_series(make_example_dataframe_series(n_channels=2))
    assert is_single_series(make_example_2d_numpy_series())
    assert not is_univariate_series(make_example_3d_numpy())
    assert not is_univariate_series(make_example_3d_numpy_list())


def test_check_series():
    """Test check_series."""
    assert isinstance(make_example_pandas_series(), pd.Series)
    assert isinstance(make_example_1d_numpy(), np.ndarray)
    assert isinstance(make_example_dataframe_series(n_channels=2), pd.DataFrame)
    with pytest.raises(ValueError, match="Input type of y should be one "):
        check_series(None)
    # check
