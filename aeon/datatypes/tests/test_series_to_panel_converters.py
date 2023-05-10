# -*- coding: utf-8 -*-
"""Testing panel converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd

from aeon.datatypes._series_as_panel import (
    convert_Panel_to_Series,
    convert_Series_to_Panel,
)
from aeon.utils._testing.collection import make_3d_test_data, make_nested_dataframe_data
from aeon.utils._testing.series import _make_series


def test_convert_numpy_series_to_panel():
    """Test output format of series-to-panel for numpy type input."""
    X_series = _make_series(n_columns=2, return_numpy=True)
    n_time, n_var = X_series.shape

    X_panel = convert_Series_to_Panel(X_series)

    assert isinstance(X_panel, np.ndarray)
    assert X_panel.ndim == 3
    assert X_panel.shape == (1, n_var, n_time)


def test_convert_numpy_panel_to_series():
    """Test output format of panel-to-series for numpy type input."""
    X_panel, _ = make_3d_test_data(n_cases=1, n_channels=2)
    _, n_var, n_time = X_panel.shape

    X_series = convert_Panel_to_Series(X_panel)

    assert isinstance(X_series, np.ndarray)
    assert X_series.ndim == 2
    assert X_series.shape == (n_time, n_var)


def test_convert_df_series_to_panel():
    """Test output format of series-to-panel for dataframe type input."""
    X_series, _ = make_nested_dataframe_data(n_cases=2)

    X_panel = convert_Series_to_Panel(X_series)

    assert isinstance(X_panel, list)
    assert isinstance(X_panel[0], pd.DataFrame)
    assert X_panel[0].equals(X_series)
