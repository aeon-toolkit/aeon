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
            20.0745,
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
            3,
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
