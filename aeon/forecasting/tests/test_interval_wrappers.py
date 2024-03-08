"""Tests the conformal interval wrapper."""

__maintainer__ = []

import numpy as np
import pandas as pd
import pytest

from aeon.datasets import load_airline
from aeon.datatypes import convert_to
from aeon.forecasting.conformal import ConformalIntervals
from aeon.forecasting.model_evaluation import evaluate
from aeon.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from aeon.forecasting.naive import NaiveForecaster, NaiveVariance
from aeon.performance_metrics.forecasting import mean_squared_error
from aeon.testing.test_config import PR_TESTING

if PR_TESTING:
    INTERVAL_WRAPPERS = [NaiveVariance]
    CV_SPLITTERS = [SlidingWindowSplitter]
    EVALUATE_STRATEGY = ["update"]
    SAMPLE_FRACS = [0.5]
    SERIES_TYPES = ["pd.Series", "pd.DataFrame"]
else:
    INTERVAL_WRAPPERS = [ConformalIntervals, NaiveVariance]
    CV_SPLITTERS = [SlidingWindowSplitter, ExpandingWindowSplitter]
    EVALUATE_STRATEGY = ["update", "refit"]
    SAMPLE_FRACS = [None, 0.5]
    SERIES_TYPES = ["pd.Series", "pd.DataFrame", "np.ndarray"]


@pytest.mark.parametrize("wrapper", INTERVAL_WRAPPERS)
@pytest.mark.parametrize("override_y_type", [True, False])
@pytest.mark.parametrize("input_type", SERIES_TYPES)
def test_wrapper_series_mtype(wrapper, override_y_type, input_type):
    """Test that interval wrappers behave nicely with different internal y_mtypes.

    The wrappers require y to be pd.Series, and the internal estimator can have
    a different internal mtype.

    We test all interval wrappers in aeon (wrapper).

    We test once with an internal forecaster that needs pd.DataFrame conversion,
    and one that accepts pd.Series.
    We do this with a trick: the vanilla NaiveForecaster can accept both; we mimick a
    "pd.DataFrame only" forecaster by restricting its y_inner_type tag to pd.Series.
    """
    y = load_airline()
    y = convert_to(y, to_type=input_type)

    f = NaiveForecaster()

    if override_y_type:
        f.set_tags(**{"y_inner_type": "pd.DataFrame"})

    interval_forecaster = wrapper(f)
    interval_forecaster.fit(y, fh=[1, 2, 3])
    pred_int = interval_forecaster.predict_interval()

    assert isinstance(pred_int, pd.DataFrame)
    assert len(pred_int) == 3

    pred_var = interval_forecaster.predict_var()

    assert isinstance(pred_var, pd.DataFrame)
    assert len(pred_var) == 3


@pytest.mark.parametrize("wrapper", INTERVAL_WRAPPERS)
@pytest.mark.parametrize("splitter", CV_SPLITTERS)
@pytest.mark.parametrize("strategy", EVALUATE_STRATEGY)
@pytest.mark.parametrize("sample_frac", SAMPLE_FRACS)
def test_evaluate_with_window_splitters(wrapper, splitter, strategy, sample_frac):
    """Test interval wrappers with different strategies and cross validators.

    The wrapper does some internal sliding window cross-validation to
    calculate the `residuals_matrix`, which means the initial cross-validation
    can cause issues.

    This checks refit and update strategies as well as expanding and sliding window
    splitters.
    """
    y = load_airline()

    if splitter == SlidingWindowSplitter:
        cv = splitter(
            fh=np.arange(1, 13),
            window_length=48,
            step_length=12,
        )
    elif splitter == ExpandingWindowSplitter:
        cv = splitter(
            fh=np.arange(1, 13),
            initial_window=48,
            step_length=12,
        )

    f = NaiveForecaster()

    if wrapper == ConformalIntervals:
        interval_forecaster = wrapper(f, initial_window=24, sample_frac=sample_frac)
    else:
        interval_forecaster = wrapper(f, initial_window=24)

    results = evaluate(
        forecaster=interval_forecaster,
        cv=cv,
        y=y,
        X=None,
        strategy=strategy,
        scoring=mean_squared_error,
        return_data=True,
        error_score="raise",
        backend=None,
    )

    assert len(results) == 8
    assert not results.test_mean_squared_error.isna().any()
