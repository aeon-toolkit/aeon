"""Error, Trend, and Seasonality (ETS) forecasters.

Implementations of the ETS forecasting algorithm with additive and multiplicative
error models. There are two versions: ETS and AutoETS. We recommend using the AutoETS
version, the configurable ETS is primarily useful for examples.
"""

__maintainer__ = []
__all__ = ["ETS", "AutoETS"]


import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._extract_paras import _extract_ets_params
from aeon.forecasting.utils._loss_functions import (
    _ets_fit,
    _ets_initialise,
    _ets_predict_value,
)
from aeon.forecasting.utils._nelder_mead import _fit_ets_parameters
from aeon.forecasting.utils._seasonality import calc_seasonal_period

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
        Type of error model: 'additive' (1) or 'multiplicative' (2)
    trend_type : string, int or None, default=0
        Type of trend component: None (0), `additive' (1) or 'multiplicative' (2)
    seasonality_type : string or None, default=0
        Type of seasonal component: None (0), `additive' (1) or 'multiplicative' (2)
    seasonal_period : int, default=1
        Number of time points in a seasonal cycle.
    iterations : int, default=200
        Maximum number of Nelder-Mead iterations per optimisation run. The
        smoothing parameters are fitted with three runs from different starting
        points, keeping the best.
    damped_trend : bool, default=True
        If True, a trend is damped and the damping parameter ``phi`` is estimated.
        If False, the trend is undamped (``phi`` fixed at 1). Ignored when there
        is no trend.

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
    likelihood_ : float
        Log-likelihood of the fitted model.
    n_timepoints_ : int
        Number of time points in the training series.

    Notes
    -----
    ``predict`` returns the forecast computed in ``fit`` and does not use the
    series passed to it. Use ``forecast(y)`` to fit and forecast a new series,
    or ``iterative_forecast`` to forecast several steps ahead.

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
    407.0178847629523
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
        damped_trend: bool = True,
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
        self.damped_trend = damped_trend
        self.n_timepoints_ = 0
        self.avg_mean_sq_err_ = 0
        self.likelihood_ = 0
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
                1 if self.damped_trend else 0,
            ],
            dtype=np.int32,
        )
        data = np.asarray(y.squeeze(), dtype=np.float64)
        self.parameters_, self.aic_ = _fit_ets_parameters(
            data, self._model, max_iter=self.iterations
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
            self.likelihood_,
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

    @property
    def liklihood_(self):
        """Deprecated misspelled alias for :attr:`likelihood_`.

        Kept for one release cycle so existing callers do not break when the
        attribute name is corrected. Slated for removal in a future release;
        switch to :attr:`likelihood_`.
        """
        import warnings

        warnings.warn(
            "ETS.liklihood_ is deprecated and will be removed in a future "
            "release; use ETS.likelihood_ instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.likelihood_

    def _predict(self, y, exog=None):
        """
        Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            Ignored. The forecast is computed in ``fit`` from the series seen
            there. Use ``forecast(y)`` to fit and forecast a new series.
        exog : np.ndarray, default =None
            Not supported.

        Returns
        -------
        float
            The prediction ``horizon`` steps ahead of the series seen in ``fit``.
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

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Forecast with ETS specific iterative method.

        Overrides the base class iterative_forecast to avoid refitting on each
        step; rolls the fitted ETS model forward in closed form. ETS does not
        support exogenous variables, so ``exog`` and ``future_exog`` are
        accepted for signature compatibility with
        :class:`~aeon.forecasting.base.IterativeForecastingMixin` only and
        raise :class:`NotImplementedError` if either is provided.
        """
        if exog is not None or future_exog is not None:
            raise NotImplementedError("ETS does not support exog.")
        y, exog, future_exog = self._check_iterative_forecast_inputs(
            y, prediction_horizon, exog, future_exog
        )
        self.fit(y)
        return self._iterative_forecast_from_fitted(prediction_horizon)

    def _iterative_forecast_from_fitted(self, prediction_horizon):
        """Forecast multiple steps ahead from fitted ETS state."""
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


class AutoETS(BaseForecaster, IterativeForecastingMixin):
    """Automatic Exponential Smoothing forecaster.

    An implementation of the exponential smoothing statistics forecasting algorithm.
    Chooses between additive and multiplicative error models, None and additive
    (damped or undamped) trend, and None, additive and multiplicative seasonality
    [1]_. Multiplicative trend is supported by :class:`ETS`, but is excluded from
    the automatic search by default because it is numerically fragile.

    There are issues relating to stability
    and efficiency discussed here https://github.com/aeon-toolkit/aeon/issues/3294.


    Parameters
    ----------
    seasonal_period : int or None, default=None
        The seasonal period to use in automatic model selection. If ``None``,
        the seasonal period is estimated from the data. If an integer is supplied,
        AutoETS selects the best seasonal model when the series length is greater
        than four times the supplied period, even if a non-seasonal candidate has
        a lower AIC. Otherwise, AutoETS falls back to non-seasonal candidates.
    allow_multiplicative_trend : bool, default=False
        Whether to include multiplicative trend models in the automatic model search.
    damped_trend : bool or None, default=None
        Which trend models to search. ``None`` considers both damped and undamped
        trends, ``True`` only damped trends and ``False`` only undamped trends.
    restrict : bool, default=True
        If True, leave out models with additive errors and multiplicative
        seasonality or a multiplicative trend, as R's ``forecast::ets`` does by
        default. Their state equations divide by the level (and trend), so a
        series whose level approaches zero can produce an exploding seasonal
        state and forecasts many orders of magnitude too large, without the AIC
        showing it [2]_.

    Attributes
    ----------
    error_type_ : int
        Selected error type: additive (1) or multiplicative (2).
    trend_type_ : int
        Selected trend type: none (0), additive (1) or multiplicative (2).
    seasonality_type_ : int
        Selected seasonality type: none (0), additive (1) or multiplicative (2).
    seasonal_period_ : int
        Seasonal period of the selected model, 1 if non-seasonal.
    damped_trend_ : bool
        Whether the selected trend is damped. False if there is no trend.
    wrapped_model_ : ETS
        The fitted :class:`ETS` model with the selected configuration.

    Notes
    -----
    ``predict`` returns the forecast computed in ``fit`` and does not use the
    series passed to it. Use ``forecast(y)`` to fit and forecast a new series,
    or ``iterative_forecast`` to forecast several steps ahead.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.
    .. [2] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and
        Practice, 2nd edition, Section 7.7. OTexts, 2018.

    Examples
    --------
    >>> from aeon.forecasting.stats import AutoETS
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoETS()
    >>> pred = forecaster.forecast(y)
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(
        self,
        seasonal_period: int | None = None,
        allow_multiplicative_trend: bool = False,
        damped_trend: bool | None = None,
        restrict: bool = True,
    ):
        self.seasonal_period = seasonal_period
        self.allow_multiplicative_trend = allow_multiplicative_trend
        self.damped_trend = damped_trend
        self.restrict = restrict
        self.error_type_ = 0
        self.trend_type_ = 0
        self.seasonality_type_ = 0
        self.seasonal_period_ = 0
        self.damped_trend_ = False
        self.wrapped_model_ = None
        super().__init__(horizon=1, axis=1)

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
            Fitted AutoETS.
        """
        data = np.asarray(y.squeeze(), dtype=np.float64)
        if self.seasonal_period is None:
            seasonal_period = 0
            require_seasonality = False
        elif self.seasonal_period < 1:
            raise ValueError(
                f"seasonal_period must be a positive integer or None, "
                f"but saw {self.seasonal_period}"
            )
        else:
            seasonal_period = self.seasonal_period
            require_seasonality = True
        if self.damped_trend is None:
            damped_option = -1
        else:
            damped_option = 1 if self.damped_trend else 0
        best_model = auto_ets(
            data,
            self.allow_multiplicative_trend,
            seasonal_period,
            require_seasonality,
            damped_option,
            self.restrict,
        )
        self.error_type_ = int(best_model[0])
        self.trend_type_ = int(best_model[1])
        self.seasonality_type_ = int(best_model[2])
        self.seasonal_period_ = int(best_model[3])
        self.damped_trend_ = self.trend_type_ != 0 and bool(best_model[4])
        self.wrapped_model_ = ETS(
            self.error_type_,
            self.trend_type_,
            self.seasonality_type_,
            self.seasonal_period_,
            damped_trend=self.damped_trend_,
        )
        self.wrapped_model_.fit(y, exog)
        self.forecast_ = self.wrapped_model_.forecast_
        return self

    def _predict(self, y=None, exog=None):
        """
        Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            Ignored. The forecast is computed in ``fit`` from the series seen
            there. Use ``forecast(y)`` to fit and forecast a new series.
        exog : np.ndarray, default =None
            Not supported.

        Returns
        -------
        float
            The prediction ``horizon`` steps ahead of the series seen in ``fit``.
        """
        return self.wrapped_model_.predict(y, exog)

    def _forecast(self, y, exog=None, axis=1):
        self.fit(y, exog=exog)
        return float(self.wrapped_model_.forecast_)

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Forecast with ETS specific iterative method.

        Overrides the base class iterative_forecast to avoid refitting on each
        step; simply delegates to the fitted wrapped ETS model. Exogenous
        variables are not supported by AutoETS, so ``exog`` and ``future_exog``
        are accepted for signature compatibility with
        :class:`~aeon.forecasting.base.IterativeForecastingMixin` only and
        raise :class:`NotImplementedError` if either is provided.
        """
        if exog is not None or future_exog is not None:
            raise NotImplementedError("AutoETS does not support exog.")
        y, exog, future_exog = self._check_iterative_forecast_inputs(
            y, prediction_horizon, exog, future_exog
        )
        self.fit(y, exog=exog)
        return self.wrapped_model_._iterative_forecast_from_fitted(prediction_horizon)


@njit(cache=True, fastmath={"contract"})
def auto_ets(
    data,
    allow_multiplicative_trend=False,
    seasonal_period=0,
    require_seasonality=False,
    damped_option=-1,
    restrict=True,
):
    """Select an ETS model by AIC, fitting each candidate with Nelder-Mead.

    ``damped_option`` restricts trend candidates: -1 tries damped and undamped
    trends, 1 only damped and 0 only undamped. ``restrict`` leaves out
    additive error models with multiplicative seasonality or trend. Returns
    ``[error, trend, seasonality, period, damped]``.
    """
    if seasonal_period < 1:
        seasonal_period = calc_seasonal_period(data)
        require_seasonality = False
    # Technically only needs to be 2 * seasonal periods to calculate initial conditions,
    # but makes no sense to run a seasonal model with any less than 2 seasonal periods
    # worth of usable data even that might be a bit low
    seasonal_enabled = seasonal_period > 1 and len(data) > seasonal_period * 4
    s_max = 3 if seasonal_enabled else 1
    all_pos = True
    for i in range(len(data)):
        if data[i] <= 0.0:
            all_pos = False
            break
    model = np.empty(5, dtype=np.int32)
    best_model = np.empty(5, dtype=np.int32)
    best_model[0] = 1
    best_model[1] = 0
    best_model[2] = 0
    best_model[3] = 1
    best_model[4] = 0
    best_seasonal_model = np.empty(5, dtype=np.int32)
    best_aic = np.inf
    best_seasonal_aic = np.inf
    t_max = 3 if allow_multiplicative_trend else 2
    for error_type in range(1, 3):
        if error_type == 2 and not all_pos:
            continue
        for trend_type in range(0, t_max):
            if trend_type == 2 and not all_pos:
                continue
            for damped in range(0, 2):
                if trend_type == 0:
                    # without a trend, damping is irrelevant: one candidate
                    if damped == 1:
                        continue
                elif damped_option != -1 and damped != damped_option:
                    continue
                for seasonality_type in range(0, s_max):
                    if seasonality_type == 2 and not all_pos:
                        continue
                    if (
                        restrict
                        and error_type == 1
                        and (seasonality_type == 2 or trend_type == 2)
                    ):
                        continue
                    model[0] = error_type
                    model[1] = trend_type
                    model[2] = seasonality_type
                    model[3] = seasonal_period if (seasonality_type != 0) else 1
                    model[4] = damped
                    _, aic = _fit_ets_parameters(data, model, max_iter=500)
                    if aic < best_aic:
                        best_aic = aic
                        best_model[:] = model
                    if seasonality_type != 0 and aic < best_seasonal_aic:
                        best_seasonal_aic = aic
                        best_seasonal_model[:] = model
    if require_seasonality and seasonal_enabled and best_seasonal_aic < np.inf:
        return best_seasonal_model
    return best_model
