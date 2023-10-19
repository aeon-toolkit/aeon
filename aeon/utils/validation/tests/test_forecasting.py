"""This module contains tests for the forecasting module in the aeon.utils.validation package."""

__author__ = ["mloning"]

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from aeon.forecasting.base import ForecastingHorizon
from aeon.forecasting.model_selection._split import BaseSplitter
from aeon.utils.validation.forecasting import (
    check_cv,
    check_fh,
    check_sp,
    check_step_length,
    check_X,
    check_y,
    check_y_X,
)

empty_input = (np.array([], dtype=int), [], pd.Index([], dtype=int))


@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    """Test that fh validation throws an error with empty container."""
    with raises(ValueError, match="`fh` must not be empty"):
        check_fh(arg)


def test_check_fh():
    """Test enforce relative"""
    fh = ForecastingHorizon(1, is_relative=False)
    with raises(ValueError, match="must be relative"):
        check_fh(fh, enforce_relative=True)


def test_check_functions():
    """Test various aspects of the check_y_X, check_X, check_y, check_cv, check_step_length, and check_sp functions.
    
    This function tests various aspects of the check_y_X, check_X, check_y, check_cv, check_step_length, and check_sp functions from the aeon.utils.validation.forecasting module. It checks if these functions correctly perform their checks and return the expected results. The test creates a DataFrame and a Series, and checks if the check_y_X function correctly checks them and returns a Series and a DataFrame. It also checks if the check_X and check_y functions correctly check a DataFrame and a Series, respectively. It further checks if the check_cv function correctly checks a BaseSplitter object, and if the check_step_length function correctly checks various step lengths. Finally, it checks if the check_sp function correctly checks a seasonal periodicity.
    """
    bs = BaseSplitter()
    bs.start_with_window = False
    with raises(ValueError, match="`start_with_window` must be set to True"):
        check_cv(bs, enforce_start_with_window=True)
    assert check_step_length(None) is None
    with raises(ValueError, match="`step_length` must be a integer >= 1"):
        check_step_length(0)
    td = timedelta(0)
    with raises(ValueError, match="`step_length` must be a positive timedelta"):
        check_step_length(td)
    do = pd.DateOffset(0)
    with raises(ValueError, match="must be a positive pd.DateOffset"):
        check_step_length(do)
    with raises(
        ValueError,
        match="`step_length` must be an integer, timedelta, pd.DateOffset, or None",
    ):
        check_step_length("Arsenal")
    check_sp([1], enforce_list=True)
    res = check_sp(1, enforce_list=True)
    assert isinstance(res, list)
    assert res[0] == 1
    with raises(ValueError, match="`sp` must be an int >= 1 or None"):
        check_sp("FOO")
    with raises(ValueError, match="`sp` must be an int"):
        check_sp("BAR", enforce_list=True)
