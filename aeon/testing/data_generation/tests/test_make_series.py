"""Test make series."""

import numpy as np
import pandas as pd

from aeon.testing.data_generation import make_series


def test_make_series():
    """Test make series."""
    y = make_series(n_timepoints=10, n_columns=1)
    assert y.shape[0] == 10
    assert isinstance(y, pd.Series)
    y = make_series(n_timepoints=10, n_columns=2)
    assert y.shape == (10, 2)
    assert isinstance(y, pd.DataFrame)
    y = make_series(n_timepoints=10, n_columns=1, return_numpy=True)
    assert isinstance(y, np.ndarray)
    y = make_series(n_timepoints=10, n_columns=2, add_nan=True, return_numpy=True)
    assert np.isnan(y).any()
    y = make_series(n_timepoints=10, n_columns=2, all_positive=False)
    assert isinstance(y, pd.DataFrame)
