"""Testing panel converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd

from aeon.datatypes._panel._convert import from_3d_numpy_to_multi_index
from aeon.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
)
from aeon.testing.utils.data_gen import make_example_3d_numpy, make_series


def test_convert_numpy_series_to_collection():
    """Test output format of series-to-panel for numpy type input."""
    X_series = make_series(n_columns=2, return_numpy=True)
    n_time, n_var = X_series.shape

    X_collection = convert_Series_to_Panel(X_series)

    assert isinstance(X_collection, np.ndarray)
    assert X_collection.ndim == 3
    assert X_collection.shape == (1, n_var, n_time)


def test_convert_numpy_collection_to_series():
    """Test output format of panel-to-series for numpy type input."""
    X_collection, _ = make_example_3d_numpy(n_cases=1, n_channels=2)
    _, n_var, n_time = X_collection.shape

    X_series = convert_Panel_to_Series(X_collection)

    assert isinstance(X_series, np.ndarray)
    assert X_series.ndim == 2
    assert X_series.shape == (n_time, n_var)


def test_convert_df_series_to_collection():
    """Test output format of series-to-panel for dataframe type input."""
    X_series = make_series(n_columns=2, return_numpy=False)
    X_collection = convert_Series_to_Panel(X_series)

    assert isinstance(X_collection, list)
    assert isinstance(X_collection[0], pd.DataFrame)
    assert X_collection[0].equals(X_series)


def test_convert_df_collection_to_series():
    """Test output format of collection-to-series for dataframe type input."""
    X_collection, _ = make_example_3d_numpy(n_cases=1, n_channels=2, n_labels=1)
    X_collection = from_3d_numpy_to_multi_index(X_collection)
    X_series = convert_Panel_to_Series(X_collection)

    assert isinstance(X_series, pd.DataFrame)
    assert len(X_series) == len(X_collection)
    assert (X_series.values == X_collection.values).all()
