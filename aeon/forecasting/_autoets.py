"""AutoETS class.

Extends the ETSForecaster to automatically calculate the smoothing parameters

"""

__maintainer__ = []
__all__ = ["AutoETSForecaster"]
import numpy as np
from numba import njit
from scipy.optimize import minimize

from aeon.forecasting._autoets_gradient_params import _calc_model_liklihood
from aeon.forecasting._ets_fast import _fit, _predict
from aeon.forecasting._utils import calc_seasonal_period
from aeon.forecasting.base import BaseForecaster

NOGIL = False
CACHE = True


class AutoETSForecaster(BaseForecaster):
    """Automatic Exponential Smoothing forecaster.

    An implementation of the exponential smoothing statistics forecasting algorithm.
    Chooses betweek additive and multiplicative error models,
    None, additive and multiplicative (including damped) trend and
    None, additive and mutliplicative seasonality[1]_.

    Parameters
    ----------
    horizon : int, default = 1
        The horizon to forecast to.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.

    Examples
    --------
    >>> from aeon.forecasting import AutoETSForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoETSForecaster()
    >>> forecaster.fit(y)
    AutoETSForecaster()
    >>> forecaster.predict()
    366.90200486015596
    """

    def __init__(
        self,
        method="internal_nelder_mead",
        horizon=1,
    ):
        self.method = method
        self.forecast_val_ = 0.0
        self.level_ = 0.0
        self.trend_ = 0.0
        self.seasonality_ = None
        self.alpha_ = 0
        self.beta_ = 0
        self.gamma_ = 0
        self.phi_ = 0
        self.error_type_ = 0
        self.trend_type_ = 0
        self.seasonality_type_ = 0
        self.seasonal_period_ = 0
        self.n_timepoints_ = 0
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        self.k_ = 0
        self.aic_ = 0
        self.residuals_ = []
        self.fitted_values_ = []
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit Auto Exponential Smoothing forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        self
            Fitted AutoETSForecaster.
        """
        data = y.squeeze()
        (
            self.error_type_,
            self.trend_type_,
            self.seasonality_type_,
            self.seasonal_period_,
            self.alpha_,
            self.beta_,
            self.gamma_,
            self.phi_,
        ) = auto_ets(data, self.method)
        (
            self.level_,
            self.trend_,
            self.seasonality_,
            self.n_timepoints_,
            self.residuals_,
            self.fitted_values_,
            self.avg_mean_sq_err_,
            self.liklihood_,
            self.k_,
            self.aic_,
        ) = _fit(
            data,
            self.error_type_,
            self.trend_type_,
            self.seasonality_type_,
            self.seasonal_period_,
            self.alpha_,
            self.beta_,
            self.gamma_,
            self.phi_,
        )
        return self

    def _predict(self, y=None, exog=None):
        """
        Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        fitted_value = _predict(
            self.trend_type_,
            self.seasonality_type_,
            self.level_,
            self.trend_,
            self.seasonality_,
            self.phi_,
            self.horizon,
            self.n_timepoints_,
            self.seasonal_period_,
        )
        if y is None:
            return np.array([fitted_value])
        else:
            return np.insert(y, 0, fitted_value)[:-1]


def auto_ets(data, method="internal_nelder_mead"):
    """Return the best ETS model based on the supplied data, and optimisation method."""
    if method == "internal_nelder_mead":
        return auto_ets_nelder_mead(data)
    elif method == "internal_gradient":
        return auto_ets_gradient(data)
    else:
        return auto_ets_scipy(data, method)


def auto_ets_scipy(data, method):
    """Calculate ETS model parameters based on scipy optimisation functions."""
    seasonal_period = calc_seasonal_period(data)
    lowest_liklihood = -1
    best_model = None
    for error_type in range(1, 3):
        for trend_type in range(0, 3):
            for seasonality_type in range(0, 2 * (seasonal_period != 1) + 1):
                optimise_result = optimise_params_scipy(
                    data,
                    error_type,
                    trend_type,
                    seasonality_type,
                    seasonal_period,
                    method,
                )
                alpha, beta, gamma = optimise_result.x
                liklihood_ = optimise_result.fun
                phi = 0.98
                if lowest_liklihood == -1 or lowest_liklihood > liklihood_:
                    lowest_liklihood = liklihood_
                    best_model = (
                        error_type,
                        trend_type,
                        seasonality_type,
                        seasonal_period,
                        alpha,
                        beta,
                        gamma,
                        phi,
                    )
    return best_model


def auto_ets_gradient(data):
    """
    Calc model params using pytorch.

    Calculate ETS model parameters based on the
    internal gradient-based approach using pytorch.
    """
    seasonal_period = calc_seasonal_period(data)
    lowest_liklihood = -1
    best_model = None
    for error_type in range(1, 3):
        for trend_type in range(0, 3):
            for seasonality_type in range(0, 2 * (seasonal_period != 1) + 1):
                (alpha, beta, gamma, phi, _residuals, liklihood_) = (
                    _calc_model_liklihood(
                        data, error_type, trend_type, seasonality_type, seasonal_period
                    )
                )
                if lowest_liklihood == -1 or lowest_liklihood > liklihood_:
                    lowest_liklihood = liklihood_
                    best_model = (
                        error_type,
                        trend_type,
                        seasonality_type,
                        seasonal_period,
                        alpha,
                        beta,
                        gamma,
                        phi,
                    )
    return best_model


@njit(nogil=NOGIL, cache=CACHE)
def auto_ets_nelder_mead(data):
    """Calculate model parameters based on the internal nelder-mead implementation."""
    seasonal_period = calc_seasonal_period(data)
    lowest_aic = -1
    best_model = None
    for error_type in range(1, 3):
        for trend_type in range(0, 3):
            for seasonality_type in range(0, 2 * (seasonal_period != 1) + 1):
                ([alpha, beta, gamma, phi], aic) = nelder_mead(
                    data, error_type, trend_type, seasonality_type, seasonal_period
                )
                if trend_type == 0:
                    phi = 1
                if lowest_aic == -1 or lowest_aic > aic:
                    lowest_aic = aic
                    best_model = (
                        error_type,
                        trend_type,
                        seasonality_type,
                        seasonal_period,
                        alpha,
                        beta,
                        gamma,
                        phi,
                    )
    return best_model


def optimise_params_scipy(
    data, error_type, trend_type, seasonality_type, seasonal_period, method
):
    """Optimise the ETS model parameters using the scipy algorithms."""

    def run_ets_scipy(parameters):
        alpha, beta, gamma, phi = parameters
        if not (
            0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and 0 <= phi <= 1
        ):
            return float("inf")
        (
            _level,
            _trend,
            _seasonality,
            _n_timepoints,
            _residuals,
            _fitted_values,
            _avg_mean_sq_err,
            _liklihood,
            _k,
            aic_,
        ) = _fit(
            data,
            error_type,
            trend_type,
            seasonality_type,
            seasonal_period,
            alpha,
            beta,
            gamma,
            phi,
        )
        return aic_

    initial_points = [0.5, 0.5, 0.5, 0.5]
    return minimize(
        run_ets_scipy, initial_points, bounds=[[0, 1] for i in range(3)], method=method
    )


@njit(nogil=NOGIL, cache=CACHE)
def run_ets(
    parameters, data, error_type, trend_type, seasonality_type, seasonal_period
):
    """Create and fit an ETS model and return the liklihood."""
    alpha, beta, gamma, phi = parameters
    if not (
        0 <= alpha <= 1
        and 0 <= beta <= 1
        and 0 <= gamma <= 1
        and 0.8 <= phi <= 1
        and (data.min() > 0 or error_type != 2)
    ):
        return np.finfo(np.float64).max
    (
        _level,
        _trend,
        _seasonality,
        _n_timepoints,
        _residuals,
        _fitted_values,
        _avg_mean_sq_err,
        _liklihood,
        _k,
        aic_,
    ) = _fit(
        data,
        error_type,
        trend_type,
        seasonality_type,
        seasonal_period,
        alpha,
        beta,
        gamma,
        phi,
    )
    return aic_


@njit(nogil=NOGIL, cache=CACHE)
def nelder_mead(
    data,
    error_type,
    trend_type,
    seasonality_type,
    seasonal_period,
    tol=1e-6,
    max_iter=500,
):
    """Implement the nelder-mead optimisation algorithm."""
    points = np.array(
        [
            [0.5, 0.5, 0.5, 0.9],
            [0.6, 0.5, 0.5, 0.9],
            [0.5, 0.6, 0.5, 0.9],
            [0.5, 0.5, 0.6, 0.9],
            [0.5, 0.5, 0.5, 0.95],
        ]
    )
    values = np.array(
        [
            run_ets(v, data, error_type, trend_type, seasonality_type, seasonal_period)
            for v in points
        ]
    )
    for _iteration in range(max_iter):
        # Order simplex by function values
        order = np.argsort(values)
        points = points[order]
        values = values[order]

        # Centroid of the best n points
        centre_point = points[:-1].sum(axis=0) / len(points[:-1])

        # Reflection
        # centre + distance between centre and largest value
        reflected_point = centre_point + (centre_point - points[-1])
        reflected_value = run_ets(
            reflected_point,
            data,
            error_type,
            trend_type,
            seasonality_type,
            seasonal_period,
        )
        # if between best and second best, use reflected value
        if values[0] <= reflected_value < values[-2]:
            points[-1] = reflected_point
            values[-1] = reflected_value
            continue
        # Expansion
        # Otherwise if it is better than the best value
        if reflected_value < values[0]:
            expanded_point = centre_point + 2 * (reflected_point - centre_point)
            expanded_value = run_ets(
                expanded_point,
                data,
                error_type,
                trend_type,
                seasonality_type,
                seasonal_period,
            )
            # if less than reflected value use expanded, otherwise go back to reflected
            if expanded_value < reflected_value:
                points[-1] = expanded_point
                values[-1] = expanded_value
            else:
                points[-1] = reflected_point
                values[-1] = reflected_value
            continue
        # Contraction
        # Otherwise if reflection is worse than all current values
        contracted_point = centre_point - 0.5 * (centre_point - points[-1])
        contracted_value = run_ets(
            contracted_point,
            data,
            error_type,
            trend_type,
            seasonality_type,
            seasonal_period,
        )
        # If contraction is better use that otherwise move to shrinkage
        if contracted_value < values[-1]:
            points[-1] = contracted_point
            values[-1] = contracted_value
            continue

        # Shrinkage
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = run_ets(
                points[i],
                data,
                error_type,
                trend_type,
                seasonality_type,
                seasonal_period,
            )

        # Convergence check
        if np.max(np.abs(values - values[0])) < tol:
            break
    return points[0], values[0]
