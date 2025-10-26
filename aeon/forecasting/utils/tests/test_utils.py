"""Forecasting utils tests."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.forecasting.utils._extract_paras import _extract_arma_params
from aeon.forecasting.utils._loss_functions import _arima_fit
from aeon.forecasting.utils._nelder_mead import nelder_mead


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
