"""Test for making forecasting problems."""

__maintainer__ = []
__all__ = []

import pandas as pd
import pytest

from aeon.testing.utils.data_gen import make_forecasting_problem


@pytest.mark.parametrize("n_timepoints", [3, 5])
def test_make_forecasting_problem(n_timepoints):
    y = make_forecasting_problem(n_timepoints)

    assert isinstance(y, pd.Series)
    assert y.shape[0] == n_timepoints
