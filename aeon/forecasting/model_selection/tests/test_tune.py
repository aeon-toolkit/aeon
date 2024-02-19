"""Test grid search CV."""

__maintainer__ = []
__all__ = ["test_gscv", "test_rscv"]

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid, ParameterSampler

from aeon.datasets import load_longley
from aeon.forecasting.compose import TransformedTargetForecaster
from aeon.forecasting.model_evaluation import evaluate
from aeon.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from aeon.forecasting.naive import NaiveForecaster
from aeon.forecasting.tests import TEST_N_ITERS, TEST_OOS_FHS, TEST_WINDOW_LENGTHS_INT
from aeon.forecasting.trend import PolynomialTrendForecaster
from aeon.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_squared_error,
)
from aeon.testing.test_config import PR_TESTING
from aeon.testing.utils.data_gen import _make_hierarchical
from aeon.transformations.detrend import Detrender

NAIVE = NaiveForecaster(strategy="mean")
NAIVE_GRID = {"window_length": TEST_WINDOW_LENGTHS_INT}
PIPE = TransformedTargetForecaster(
    [
        ("transformer", Detrender(PolynomialTrendForecaster())),
        ("forecaster", NaiveForecaster()),
    ]
)
PIPE_GRID = {
    "transformer__forecaster__degree": [1, 2],
    "forecaster__strategy": ["last", "mean"],
}

if PR_TESTING:
    TEST_METRICS = [mean_absolute_percentage_error]
    ERROR_SCORES = [1000]
    GRID = [(NAIVE, NAIVE_GRID)]
else:
    TEST_METRICS = [mean_absolute_percentage_error, mean_squared_error]
    ERROR_SCORES = [np.nan, "raise", 1000]
    GRID = [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]

CVs = [
    *[SingleWindowSplitter(fh=fh) for fh in TEST_OOS_FHS],
    SlidingWindowSplitter(fh=1, initial_window=15),
]


def _get_expected_scores(forecaster, cv, param_grid, y, X, scoring):
    scores = np.zeros(len(param_grid))
    for i, params in enumerate(param_grid):
        f = forecaster.clone()
        f.set_params(**params)
        out = evaluate(f, cv, y, X=X, scoring=scoring)
        scores[i] = out.loc[:, f"test_{scoring.__name__}"].mean()
    return scores


def _check_cv(forecaster, tuner, cv, param_grid, y, X, scoring):
    actual = tuner.cv_results_[f"mean_test_{scoring.__name__}"]

    expected = _get_expected_scores(forecaster, cv, param_grid, y, X, scoring)
    np.testing.assert_array_equal(actual, expected)

    # Check if best parameters are selected.
    best_idx = tuner.best_index_
    assert best_idx == actual.argmin()

    fitted_params = tuner.get_fitted_params()
    assert param_grid[best_idx].items() <= fitted_params.items()


@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
def test_gscv(forecaster, param_grid, cv, scoring, error_score):
    """Test ForecastingGridSearchCV."""
    y, X = load_longley()
    gscv = ForecastingGridSearchCV(
        forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
    )
    gscv.fit(y, X)

    param_grid = ParameterGrid(param_grid)
    _check_cv(forecaster, gscv, cv, param_grid, y, X, scoring)

    fitted_params = gscv.get_fitted_params()
    assert "best_forecaster" in fitted_params.keys()
    assert "best_score" in fitted_params.keys()


@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("n_iter", TEST_N_ITERS)
def test_rscv(forecaster, param_grid, cv, scoring, error_score, n_iter):
    """Test ForecastingRandomizedSearchCV.

    Tests that ForecastingRandomizedSearchCV successfully searches the
    parameter distributions to identify the best parameter set
    """
    y, X = load_longley()
    rscv = ForecastingRandomizedSearchCV(
        forecaster,
        param_distributions=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
        n_iter=n_iter,
        random_state=42,
    )
    rscv.fit(y, X)

    param_distributions = list(ParameterSampler(param_grid, n_iter, random_state=42))
    _check_cv(forecaster, rscv, cv, param_distributions, y, X, scoring)


@pytest.mark.parametrize("forecaster, param_grid", GRID)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
def test_gscv_hierarchical(forecaster, param_grid, cv, scoring, error_score):
    """Test ForecastingGridSearchCV."""
    y = _make_hierarchical(
        random_state=0, hierarchy_levels=(2, 2), min_timepoints=20, max_timepoints=20
    )
    X = _make_hierarchical(
        random_state=42, hierarchy_levels=(2, 2), min_timepoints=20, max_timepoints=20
    )

    gscv = ForecastingGridSearchCV(
        forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
    )
    gscv.fit(y, X)

    param_grid = ParameterGrid(param_grid)
    _check_cv(forecaster, gscv, cv, param_grid, y, X, scoring)
