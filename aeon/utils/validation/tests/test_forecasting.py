"""Test forecasting module."""

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
    X = pd.DataFrame(np.random.randint(0, 10, size=(6, 4)), columns=list("ABCD"))
    y = pd.Series([1, 2, 3, 4, 5, 6])
    y2, X2 = check_y_X(y, X)
    assert isinstance(y2, pd.Series)
    assert isinstance(X2, pd.DataFrame)
    X2 = check_X(X)
    assert isinstance(X2, pd.DataFrame)
    y2 = check_y(y)
    assert isinstance(y2, pd.Series)
    y = pd.Series([1, 1, 1, 1, 1])
    with raises(ValueError, match="All values of `y` are the same"):
        check_y(y, allow_constant=False)
    with raises(TypeError, match=f"`cv` is not an instance of {BaseSplitter}"):
        check_cv("Arsenal")
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
