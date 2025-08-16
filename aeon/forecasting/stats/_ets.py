"""ETS class.

An implementation of the exponential smoothing statistics forecasting algorithm.
Implements additive and multiplicative error models. We recommend using the AutoETS
version, but this is useful for demonstrations.
"""

__maintainer__ = []
__all__ = ["ETS"]


import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._extract_paras import _extract_ets_params
from aeon.forecasting.utils._loss_functions import (
    _ets_fit,
    _ets_initialise,
    _ets_predict_value,
)
from aeon.forecasting.utils._nelder_mead import nelder_mead

ADDITIVE = "additive"
MULTIPLICATIVE = "multiplicative"


class ETS(BaseForecaster, IterativeForecastingMixin):
    """Exponential Smoothing (ETS) forecaster.

    Implements the ETS (Error, Trend, Seasonality) forecaster, supporting additive
    and multiplicative forms of error, trend (including damped), and seasonality
    components. Based on the state space model formulation of exponential
    smoothing as described in Hyndman and Athanasopoulos [1]_.

    Parameters
    ----------
    error_type : string or int, default=1
        Type of error model: 'additive' (0) or 'multiplicative' (1)
    trend_type : string, int or None, default=0
        Type of trend component: None (0), `additive' (1) or 'multiplicative' (2)
    seasonality_type : string or None, default=0
        Type of seasonal component: None (0), `additive' (1) or 'multiplicative' (2)
    seasonal_period : int, default=1
        Number of time points in a seasonal cycle.
    alpha : float, default=0.1
        Level smoothing parameter.
    beta : float, default=0.01
        Trend smoothing parameter.
    gamma : float, default=0.01
        Seasonal smoothing parameter.
    phi : float, default=0.99
        Trend damping parameter (used only for damped trend models).

    Attributes
    ----------
    forecast_val_ : float
        Forecast value for the given horizon.
    level_ : float
        Estimated level component.
    trend_ : float
        Estimated trend component.
    seasonality_ : array-like or None
        Estimated seasonal components.
    aic_ : float
        Akaike Information Criterion of the fitted model.
    avg_mean_sq_err_ : float
        Average mean squared error of the fitted model.
    residuals_ : list of float
        Residuals from the fitted model.
    fitted_values_ : list of float
        Fitted values for the training data.
    liklihood_ : float
        Log-likelihood of the fitted model.
    n_timepoints_ : int
        Number of time points in the training series.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice, 2nd Edition. OTexts, 2014.
       https://otexts.com/fpp3/

    Examples
    --------
    >>> from aeon.forecasting.stats import ETS
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ETS(
    ...     error_type='additive', trend_type='multiplicative',
    ...     seasonality_type='multiplicative', seasonal_period=4
    ... )
    >>> forecaster.forecast(y)
    413.07266877621925
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(
        self,
        error_type: int | str = 1,
        trend_type: int | str | None = 0,
        seasonality_type: int | str | None = 0,
        seasonal_period: int = 1,
        iterations: int = 200,
    ):
        self.forecast_val_ = 0.0
        self.level_ = 0.0
        self.trend_ = 0.0
        self.seasonality_ = None
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.seasonal_period = seasonal_period
        self.iterations = iterations
        self.n_timepoints_ = 0
        self.avg_mean_sq_err_ = 0
        self.liklihood_ = 0
        self.k_ = 0
        self.aic_ = 0
        self.residuals_ = []
        self.fitted_values_ = []
        self._model = []
        self.parameters_ = []
        self.alpha_ = 0
        self.beta_ = 0
        self.gamma_ = 0
        self.phi_ = 0
        self.forecast_ = 0
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit Exponential Smoothing forecaster to series y.

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
            Fitted ETS.
        """
        _validate_parameter(self.error_type, False)
        _validate_parameter(self.seasonality_type, True)
        _validate_parameter(self.trend_type, True)

        # Convert to string parameters to ints for numba efficiency
        def _get_int(x):
            if x is None:
                return 0
            if x == ADDITIVE:
                return 1
            if x == MULTIPLICATIVE:
                return 2
            return x

        self._error_type = _get_int(self.error_type)
        self._seasonality_type = _get_int(self.seasonality_type)
        self._trend_type = _get_int(self.trend_type)
        self._seasonal_period = self.seasonal_period
        if self._seasonal_period < 1 or self._seasonality_type == 0:
            self._seasonal_period = 1
        self._model = np.array(
            [
                self._error_type,
                self._trend_type,
                self._seasonality_type,
                self._seasonal_period,
            ],
            dtype=np.int32,
        )
        data = y.squeeze()
        (self.parameters_, self.aic_) = nelder_mead(
            1,
            1 + 2 * (self._trend_type != 0) + (self._seasonality_type != 0),
            data,
            self._model,
            max_iter=self.iterations,
        )
        self.alpha_, self.beta_, self.gamma_, self.phi_ = _extract_ets_params(
            self.parameters_, self._model
        )
        (
            self.aic_,
            self.level_,
            self.trend_,
            self.seasonality_,
            self.n_timepoints_,
            self.residuals_,
            self.fitted_values_,
            self.avg_mean_sq_err_,
            self.liklihood_,
            self.k_,
        ) = _ets_fit(self.parameters_, data, self._model)
        self.forecast_ = _numba_predict(
            self._trend_type,
            self._seasonality_type,
            self.level_,
            self.trend_,
            self.seasonality_,
            self.phi_,
            self.horizon,
            self.n_timepoints_,
            self._seasonal_period,
        )

        return self

    def _predict(self, y, exog=None):
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
        return self.forecast_

    def _initialise(self, data):
        """
        Initialize level, trend, and seasonality values for the ETS model.

        Parameters
        ----------
        data : array-like
            The time series data
            (should contain at least two full seasons if seasonality is specified)
        """
        self.level_, self.trend_, self.seasonality_ = _ets_initialise(
            self._trend_type, self._seasonality_type, self._seasonal_period, data
        )

    def iterative_forecast(self, y, prediction_horizon):
        """Forecast with ETS specific iterative method.

        Overrides the base class iterative_forecast to avoid refitting on each step.
        This simply rolls the ETS model forward
        """
        self.fit(y)
        preds = np.zeros(prediction_horizon)
        preds[0] = self.forecast_
        for i in range(1, prediction_horizon):
            preds[i] = _numba_predict(
                self._trend_type,
                self._seasonality_type,
                self.level_,
                self.trend_,
                self.seasonality_,
                self.phi_,
                i + 1,
                self.n_timepoints_,
                self._seasonal_period,
            )
        return preds


@njit(fastmath=True, cache=True)
def _numba_predict(
    trend_type,
    seasonality_type,
    level,
    trend,
    seasonality,
    phi,
    horizon,
    n_timepoints,
    seasonal_period,
):
    # Generate forecasts based on the final values of level, trend, and seasonals
    if phi == 1:  # No damping case
        phi_h = horizon
    else:
        # Geometric series formula for calculating phi + phi^2 + ... + phi^h
        phi_h = phi * (1 - phi**horizon) / (1 - phi)
    seasonal_index = (n_timepoints + horizon - 1) % seasonal_period
    return _ets_predict_value(
        trend_type,
        seasonality_type,
        level,
        trend,
        seasonality[seasonal_index],
        phi_h,
    )[0]


def _validate_parameter(var, can_be_none):
    valid_str = (ADDITIVE, MULTIPLICATIVE)
    valid_int = (1, 2)
    if can_be_none:
        valid_str = (None, ADDITIVE, MULTIPLICATIVE)
        valid_int = (0, 1, 2)
    valid = True
    if isinstance(var, str) or var is None:
        if var not in valid_str:
            valid = False
    elif isinstance(var, int):
        if var not in valid_int:
            valid = False
    else:
        valid = False
    if not valid:
        raise ValueError(
            f"variable must be either string or integer with values"
            f" {valid_str} or {valid_int} but saw {var}"
        )
