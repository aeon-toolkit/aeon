"""Forecasting utils tests."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.forecasting.utils._extract_paras import _extract_arma_params
from aeon.forecasting.utils._loss_functions import _arima_fit
from aeon.forecasting.utils._nelder_mead import dispatch_loss, nelder_mead


@pytest.mark.parametrize(
    "params, data, model, expected_aic",
    [
        (
            np.array([0.5, -0.1, 0.1]),
            np.array([1.0, 2.0, 1.5, 1.7, 2.1]),
            np.array([1, 1, 1]),
            23.5931,
        ),
        (np.array([0.0]), np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 0]), 13.3515),
    ],
)
def test_arima_fit(params, data, model, expected_aic):
    """Test ARIMA model fitting and AIC calculation."""
    result = _arima_fit(params, data, model)
    assert np.isclose(
        result, expected_aic, atol=1e-4
    ), f"AIC mismatch. Got {result}, expected {expected_aic}"


# @pytest.mark.parametrize(
#     "fn_id, params, data, model, expected_result",
#     [
#         (
#             0,
#             np.array([0.5, -0.1, 0.1]),
#             np.array([1.0, 2.0, 1.5, 1.7, 2.1]),
#             np.array([1, 1, 1]),
#             19.99880,  # example expected result from _arima_fit
#         ),
#         (
#             1,
#             np.array([0.5, 0.3, 1, 0.4]),
#             np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]),
#             np.array([1, 1, 1, 4]),
#             55.58355126510806,
#         ),
#         (
#             1,
#             np.array([0.7, 0.6, 0.97, 0.1]),
#             np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]),
#             np.array([2, 1, 1, 4]),
#             61.797186036891276,
#         ),
#         (
#             1,
#             np.array([0.4, 0.2, 0.8, 0.5]),
#             np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]),
#             np.array([1, 2, 2, 4]),
#             76.86950158342418,
#         ),
#         (
#             1,
#             np.array([0.7, 0.5, 0.85, 0.2]),
#             np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]),
#             np.array([2, 2, 2, 4]),
#             82.83246015454237,
#         ),
#         (
#             2,
#             np.array([0.0]),
#             np.array([1.0, 1.0, 1.0, 1.0]),
#             np.array([0, 0, 0]),
#             ValueError,  # expected error for unknown fn_id
#         ),
#     ],
# )
# def test_dispatch_loss(fn_id, params, data, model, expected_result):
#     """Test dispatching loss functions by function ID."""
#     if isinstance(expected_result, type) and issubclass(expected_result, Exception):
#         with pytest.raises(expected_result):
#             dispatch_loss(fn_id, params, data, model)
#     else:
#         result = dispatch_loss(fn_id, params, data, model)
#         assert np.isclose(
#             result, expected_result, atol=1e-4
#         ), f"Result mismatch. Got {result}, expected {expected_result}"


@pytest.mark.parametrize(
    "params, model, expected_result",
    [
        (
            np.array([0.5, -0.3, 0.2, 0.1]),
            np.array([2, 1, 1]),
            np.array([[0.5, -0.3], [0.2, np.nan], [0.1, np.nan]]),
        ),
        (np.array([]), np.array([0, 0, 0]), np.array([], dtype=float).reshape(3, 0)),
        (np.array([0.1]), np.array([1]), np.array([[0.1]])),
    ],
)
def test_extract_arma_params(params, model, expected_result):
    """Test parameter extraction for ARMA models."""
    result = _extract_arma_params(params, model)

    assert result.shape == expected_result.shape, "Output shape mismatch."
    assert_array_almost_equal(result, expected_result, decimal=6)


@pytest.mark.parametrize(
    "loss_id, num_params, data, model, tol, max_iter",
    [(0, 3, np.array([1.0, 2.0, 1.5, 1.7, 2.1]), np.array([1, 1, 1]), 1e-6, 500)],
)
def test_nelder_mead(loss_id, num_params, data, model, tol, max_iter):
    """Test Nelder-Mead simplex optimisation."""
    best_params, best_value = nelder_mead(
        loss_id, num_params, data, model, tol, max_iter
    )
    assert len(best_params) == num_params, "Incorrect number of parameters returned."
    assert isinstance(best_value, float), "Best value should be a float."
    assert not np.isnan(best_value), "Best value should not be NaN."


@pytest.mark.parametrize(
    "params, model",
    [
        (np.array([-0.1]), np.array([1, 0, 0, 1], dtype=np.int32)),
        (np.array([1.0]), np.array([1, 0, 0, 1], dtype=np.int32)),
        (np.array([0.5, 0.5, 0.98]), np.array([1, 1, 0, 1], dtype=np.int32)),
        (np.array([0.5, 0.1, 0.0]), np.array([1, 1, 0, 1], dtype=np.int32)),
        (np.array([0.5, 0.5]), np.array([1, 0, 1, 1], dtype=np.int32)),
    ],
)
def test_dispatch_loss_rejects_invalid_ets_params(params, model):
    """Invalid finite ETS parameters should be rejected by the optimiser."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = dispatch_loss(1, params, data, model)

    assert np.isinf(result)


def test_kpss_test_trend_and_invalid_regression():
    """KPSS supports trend stationarity and rejects unknown regression."""
    from aeon.forecasting.utils._hypo_tests import kpss_test

    rng = np.random.default_rng(0)
    y = 0.5 * np.arange(100.0) + rng.standard_normal(100)
    stat_ct, stationary_ct = kpss_test(y, regression="ct")
    assert np.isfinite(stat_ct)
    stat_lag, _ = kpss_test(y, lags=5)
    assert np.isfinite(stat_lag)
    with pytest.raises(ValueError, match="regression must be"):
        kpss_test(y, regression="bad")


def test_acf_degenerate_variance_branches():
    """Zero-variance segments return 1 (both) or 0 (one side)."""
    from aeon.forecasting.utils._seasonality import acf

    # both segments constant -> 1.0
    out = acf(np.ones(20), max_lag=3)
    assert np.allclose(out, 1.0)
    # first segment constant, second varying at lag 10 -> 0.0
    X = np.concatenate([np.ones(10), np.arange(10.0)])
    out2 = acf(X, max_lag=10)
    assert out2[9] == 0.0


def test_comb_edge_cases():
    """Binomial coefficient helper returns 0 outside the valid range."""
    from aeon.forecasting.utils._undifference import _comb

    assert _comb(3, 5) == 0
    assert _comb(3, -1) == 0
    assert _comb(5, 4) == 5


def test_dispatch_loss_unknown_id_raises():
    """An unknown loss function id raises a ValueError."""
    with pytest.raises(ValueError):
        dispatch_loss(
            2,
            np.array([0.5]),
            np.arange(10.0),
            np.array([1, 0, 0, 1], dtype=np.int32),
        )


def test_ets_state_and_cycle_validity_guards():
    """Degenerate states are rejected by the ETS validity helpers."""
    from aeon.forecasting.utils._loss_functions import (
        _ets_forecast_cycle_is_valid,
        _ets_state_is_valid,
    )

    # non-finite state
    assert not _ets_state_is_valid(1, 0, 0, np.nan, 0.0, 1.0, 1.0, 0.0)
    # multiplicative trend with non-positive level
    assert not _ets_state_is_valid(1, 2, 0, -1.0, 1.0, 1.0, 1.0, 0.0)
    # multiplicative error with non-positive forecast over the cycle
    assert not _ets_forecast_cycle_is_valid(2, 0, 0, -5.0, 0.0, np.zeros(1), 1.0, 10, 1)


def test_ets_fit_degenerate_data_returns_large_loss():
    """Invalid initial states and overflowing errors return LARGE_LOSS."""
    from aeon.forecasting.utils._loss_functions import _ets_fit

    # multiplicative error on negative data -> invalid initial state
    model = np.array([2, 0, 0, 1], dtype=np.int32)
    aic = _ets_fit(np.array([0.5]), -np.ones(10), model)[0]
    assert aic >= 1e10
    # huge alternating values overflow the squared error sum
    model_add = np.array([1, 0, 0, 1], dtype=np.int32)
    data = np.array([1e200, -1e200] * 5)
    aic2 = _ets_fit(np.array([0.5]), data, model_add)[0]
    assert aic2 >= 1e10


@pytest.mark.parametrize(
    "loss_id, model, points",
    [
        (
            0,
            np.array([1, 1, 1]),
            np.array(
                [
                    [0.2, -0.1, 0.1],
                    [0.4, 0.0, 0.2],
                    [0.1, 0.2, -0.2],
                    [0.3, 0.1, 0.0],
                ]
            ),
        ),
        (
            1,
            np.array([1, 1, 1, 4]),
            np.array(
                [
                    [0.2, 0.1, 0.9, 0.1],
                    [0.4, 0.2, 0.95, 0.2],
                    [0.1, 0.05, 0.8, 0.05],
                    [0.3, 0.15, 0.85, 0.1],
                    [0.5, 0.1, 0.98, 0.2],
                ]
            ),
        ),
    ],
)
def test_nelder_mead_from_custom_simplex(loss_id, model, points):
    """With no iterations, return the best supplied vertex without changing it."""
    from aeon.forecasting.utils._nelder_mead import _nelder_mead_from_simplex

    data = np.array([1.0, 2.0, 1.5, 1.7, 2.1, 1.9, 2.4, 2.2, 2.0, 2.6, 2.3, 2.8])
    start = points.copy()
    values = np.array(
        [dispatch_loss(loss_id, point.copy(), data, model) for point in points]
    )
    assert np.all(np.isfinite(values))

    best_params, best_value = _nelder_mead_from_simplex(
        loss_id, points, data, model, max_iter=0
    )

    best_vertex = np.argmin(values)
    assert np.array_equal(best_params, start[best_vertex])
    assert best_value == values[best_vertex]
    assert np.array_equal(points, start)


def _trending_seasonal(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    return 50.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n)


def test_fit_ets_parameters_reaches_optimum():
    """The multi-start fit escapes a local minimum a single run stops in.

    For ETS(M,A,M) on this series, one Nelder-Mead run from the default simplex
    stops ~100 AIC above the optimum. Compare against scipy's Nelder-Mead from
    several starting points.
    """
    from scipy.optimize import minimize

    from aeon.forecasting.utils._loss_functions import _ets_fit
    from aeon.forecasting.utils._nelder_mead import _fit_ets_parameters

    data = _trending_seasonal()
    model = np.array([2, 1, 2, 12, 1], dtype=np.int32)
    _, value = _fit_ets_parameters(data, model)

    def loss(x):
        v = _ets_fit(np.asarray(x, dtype=np.float64), data, model)[0]
        return v if np.isfinite(v) else 1e300

    ref = min(
        minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 20000}).fun
        for x0 in (
            [0.1, 0.01, 0.98, 0.05],
            [0.3, 0.05, 0.9, 0.2],
            [0.02, 0.001, 1.0, 0.7],
        )
    )
    assert value <= ref + 0.5


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "model",
    [[1, 1, 0, 1, 0], [1, 1, 0, 1, 1], [1, 0, 1, 12, 0], [2, 1, 2, 12, 1]],
)
def test_fit_ets_parameters_not_worse_than_single_run(seed, model):
    """Keeping the best of several runs never loses to the single default run."""
    from aeon.forecasting.utils._extract_paras import _ets_num_params
    from aeon.forecasting.utils._nelder_mead import _fit_ets_parameters

    data = _trending_seasonal(seed=seed)
    model = np.array(model, dtype=np.int32)
    _, single = nelder_mead(1, _ets_num_params(model), data, model, max_iter=200)
    _, multi = _fit_ets_parameters(data, model, max_iter=200)
    assert multi <= single


@pytest.mark.parametrize(
    "x0, model",
    [
        ([0.95, 0.9, 0.99, 0.04], [1, 1, 1, 12, 1]),
        ([0.2, 0.02, 0.978, 0.04], [1, 1, 1, 12, 1]),
        ([0.6, 0.39], [1, 0, 1, 12, 0]),
        ([0.3, 0.29, 0.5, 0.69], [1, 1, 1, 12, 1]),
        ([0.0, 0.0], [1, 1, 0, 1, 0]),  # invalid start on a bound
    ],
)
def test_ets_simplex_around_stays_valid(x0, model):
    """Every vertex of a restart simplex is inside the valid parameter region."""
    from aeon.forecasting.utils._extract_paras import _extract_ets_params
    from aeon.forecasting.utils._loss_functions import _ets_params_are_valid
    from aeon.forecasting.utils._nelder_mead import _ets_simplex_around

    model = np.array(model, dtype=np.int32)
    x0 = np.array(x0)
    points = _ets_simplex_around(x0, model)
    assert points.shape == (len(x0) + 1, len(x0))
    assert np.allclose(points[0], x0)
    # the simplex is not degenerate: each vertex differs from x0 in one place
    assert np.all(np.sum(points[1:] != points[0], axis=1) == 1)
    a, b, g, phi = _extract_ets_params(x0, model)
    if _ets_params_are_valid(a, b, g, phi, model[1], model[2]):
        for x in points[1:]:
            a, b, g, phi = _extract_ets_params(x, model)
            assert _ets_params_are_valid(a, b, g, phi, model[1], model[2])


def test_ets_r_style_start_layout():
    """The R-style start follows the parameter layout of the model."""
    from aeon.forecasting.utils._nelder_mead import _ets_r_style_start

    assert np.allclose(
        _ets_r_style_start(np.array([1, 1, 1, 12, 1])), [0.2, 0.02, 0.978, 0.04]
    )
    assert np.allclose(
        _ets_r_style_start(np.array([1, 1, 1, 12, 0])), [0.2, 0.02, 0.04]
    )
    assert np.allclose(_ets_r_style_start(np.array([1, 0, 0, 1])), [0.2])


def test_simplex_converged():
    """Convergence needs close values and close vertices, at a finite best."""
    from aeon.forecasting.utils._nelder_mead import _simplex_converged

    tight = np.array([[1.0, 2.0], [1.0 + 1e-8, 2.0], [1.0, 2.0 - 1e-8]])
    assert _simplex_converged(tight, np.array([5.0, 5.0 + 1e-8, 5.0 + 1e-7]), 1e-6)
    # values close, but the vertices are spread out
    spread = np.array([[1.0, 2.0], [1.5, 2.0], [1.0, 2.5]])
    assert not _simplex_converged(spread, np.array([5.0, 5.0, 5.0]), 1e-6)
    # vertices close, but the values are not
    assert not _simplex_converged(tight, np.array([5.0, 5.1, 5.2]), 1e-6)
    # a non-finite best value never counts as converged
    assert not _simplex_converged(tight, np.array([np.inf, np.inf, np.inf]), 1e-6)
