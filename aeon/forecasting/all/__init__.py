# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Import all time series forecasting functionality available in aeon."""

__author__ = ["mloning", "fkiraly"]


import numpy as np
import pandas as pd

from aeon.datasets import load_airline, load_longley, load_lynx, load_shampoo_sales
from aeon.forecasting.base import ForecastingHorizon
from aeon.forecasting.model_evaluation import evaluate
from aeon.forecasting.model_selection import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from aeon.performance_metrics.forecasting import (
    GeometricMeanRelativeAbsoluteError,
    GeometricMeanRelativeSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanAsymmetricError,
    MeanRelativeAbsoluteError,
    MeanSquaredError,
    MeanSquaredPercentageError,
    MeanSquaredScaledError,
    MedianAbsoluteError,
    MedianAbsolutePercentageError,
    MedianAbsoluteScaledError,
    MedianRelativeAbsoluteError,
    MedianSquaredError,
    MedianSquaredPercentageError,
    MedianSquaredScaledError,
    RelativeLoss,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    make_forecasting_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_asymmetric_error,
    mean_relative_absolute_error,
    mean_squared_error,
    mean_squared_percentage_error,
    mean_squared_scaled_error,
    median_absolute_error,
    median_absolute_percentage_error,
    median_absolute_scaled_error,
    median_relative_absolute_error,
    median_squared_error,
    median_squared_percentage_error,
    median_squared_scaled_error,
    relative_loss,
)
from aeon.registry import all_estimators
from aeon.transformations.series.detrend import Deseasonalizer, Detrender
from aeon.utils.plotting import plot_series

est_tuples = all_estimators(estimator_types="forecaster", return_names=True)
est_names, ests = zip(*est_tuples)

for i, x in enumerate(est_tuples):
    exec(f"{x[0]} = ests[{i}]")

__all__ = list(est_names) + [
    "ForecastingHorizon",
    "load_lynx",
    "load_longley",
    "load_airline",
    "load_shampoo_sales",
    "CutoffSplitter",
    "SlidingWindowSplitter",
    "SingleWindowSplitter",
    "ExpandingWindowSplitter",
    "temporal_train_test_split",
    "Deseasonalizer",
    "Detrender",
    "pd",
    "np",
    "plot_series",
    "evaluate",
    "make_forecasting_scorer",
    "MeanAbsoluteScaledError",
    "MedianAbsoluteScaledError",
    "MeanSquaredScaledError",
    "MedianSquaredScaledError",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "MedianAbsoluteError",
    "MedianSquaredError",
    "MeanAbsolutePercentageError",
    "MedianAbsolutePercentageError",
    "MeanSquaredPercentageError",
    "MedianSquaredPercentageError",
    "MeanRelativeAbsoluteError",
    "MedianRelativeAbsoluteError",
    "GeometricMeanRelativeAbsoluteError",
    "GeometricMeanRelativeSquaredError",
    "MeanAsymmetricError",
    "RelativeLoss",
    "mean_absolute_scaled_error",
    "median_absolute_scaled_error",
    "mean_squared_scaled_error",
    "median_squared_scaled_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "median_squared_error",
    "mean_absolute_percentage_error",
    "median_absolute_percentage_error",
    "mean_squared_percentage_error",
    "median_squared_percentage_error",
    "mean_relative_absolute_error",
    "median_relative_absolute_error",
    "geometric_mean_relative_absolute_error",
    "geometric_mean_relative_squared_error",
    "relative_loss",
    "mean_asymmetric_error",
]
