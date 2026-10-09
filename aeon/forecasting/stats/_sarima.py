"""Seasonal ARIMA (SARIMA) forecaster."""

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.stats._arima import _exog_as_timepoint_rows
from aeon.forecasting.utils._loss_functions import LOG_2PI
from aeon.forecasting.utils._nelder_mead import nelder_mead


def _get_active_lags(p, q, P, Q, m):
    ar_lags = set()
    for i in range(p + 1):
        for j in range(P + 1):
            lag = i + j * m
            if lag >= 1:
                ar_lags.add(lag)
    ar_lags = sorted(list(ar_lags))

    ma_lags = set()
    for i in range(q + 1):
        for j in range(Q + 1):
            lag = i + j * m
            if lag >= 1:
                ma_lags.add(lag)
    ma_lags = sorted(list(ma_lags))

    return np.array(ar_lags, dtype=np.int32), np.array(ma_lags, dtype=np.int32)


def _get_diff_poly(d, D, s):
    # Non-seasonal differencing polynomial (1 - B)^d
    poly_d = np.array([1.0], dtype=np.float64)
    for _ in range(d):
        poly_d = np.convolve(poly_d, np.array([1.0, -1.0], dtype=np.float64))

    # Seasonal differencing polynomial (1 - B^s)^D
    poly_D = np.array([1.0], dtype=np.float64)
    if D > 0 and s > 0:
        step = np.zeros(s + 1, dtype=np.float64)
        step[0] = 1.0
        step[s] = -1.0
        for _ in range(D):
            poly_D = np.convolve(poly_D, step)

    return np.convolve(poly_d, poly_D)


@njit(cache=True, fastmath=True)
def _extract_sarima_params(params, model):
    use_constant = model[0]
    p = model[1]
    q = model[2]
    P = model[3]
    Q = model[4]

    c = params[0] if use_constant else 0.0
    idx = 1 if use_constant else 0

    phi = params[idx : idx + p]
    idx += p

    theta = params[idx : idx + q]
    idx += q

    Phi = params[idx : idx + P]
    idx += P

    Theta = params[idx : idx + Q]

    return c, phi, theta, Phi, Theta


@njit(cache=True, fastmath=True)
def _sarima_model(params, data, model):
    _ = model[0]
    p = model[1]
    q = model[2]
    P = model[3]
    Q = model[4]
    m = model[5]
    n_ar_lags = model[6]
    n_ma_lags = model[7]

    ar_lags = model[8 : 8 + n_ar_lags]
    ma_lags = model[8 + n_ar_lags : 8 + n_ar_lags + n_ma_lags]

    c, phi, theta, Phi, Theta = _extract_sarima_params(params, model)

    ar_values = np.zeros(n_ar_lags)
    poly_ar_ns = np.zeros(p + 1)
    poly_ar_ns[0] = 1.0
    if p > 0:
        poly_ar_ns[1:] = -phi

    poly_ar_s = np.zeros(P * m + 1)
    poly_ar_s[0] = 1.0
    for j in range(P):
        poly_ar_s[(j + 1) * m] = -Phi[j]

    # Convolve
    na = len(poly_ar_ns)
    nb = len(poly_ar_s)
    poly_ar_comb = np.zeros(na + nb - 1)
    for i in range(na):
        for j in range(nb):
            poly_ar_comb[i + j] += poly_ar_ns[i] * poly_ar_s[j]

    for i in range(n_ar_lags):
        ar_values[i] = -poly_ar_comb[ar_lags[i]]

    ma_values = np.zeros(n_ma_lags)
    poly_ma_ns = np.zeros(q + 1)
    poly_ma_ns[0] = 1.0
    if q > 0:
        poly_ma_ns[1:] = theta

    poly_ma_s = np.zeros(Q * m + 1)
    poly_ma_s[0] = 1.0
    for j in range(Q):
        poly_ma_s[(j + 1) * m] = Theta[j]

    na = len(poly_ma_ns)
    nb = len(poly_ma_s)
    poly_ma_comb = np.zeros(na + nb - 1)
    for i in range(na):
        for j in range(nb):
            poly_ma_comb[i + j] += poly_ma_ns[i] * poly_ma_s[j]

    for i in range(n_ma_lags):
        ma_values[i] = poly_ma_comb[ma_lags[i]]

    n = len(data)
    num_predictions = n + 1
    residuals = np.zeros(num_predictions - 1)
    fitted_values = np.zeros(num_predictions)
    max_ar_lag = ar_lags[-1] if n_ar_lags > 0 else 0
    max_ma_lag = ma_lags[-1] if n_ma_lags > 0 else 0
    start = max(max_ar_lag, max_ma_lag)

    for t in range(start, num_predictions):
        ar_term = 0.0
        for j in range(n_ar_lags):
            ar_term += ar_values[j] * data[t - ar_lags[j]]

        ma_term = 0.0
        for j in range(n_ma_lags):
            ma_term += ma_values[j] * residuals[t - ma_lags[j]]

        fitted_values[t] = c + ar_term + ma_term
        if t != num_predictions - 1:
            residuals[t] = data[t] - fitted_values[t]

    sse = 0.0
    for i in range(start, n):
        sse += residuals[i] * residuals[i]
    variance = sse / (n - start) if n - start > 0 else 0.0
    likelihood = (
        (n - start) * (LOG_2PI + np.log(variance) + 1.0) if n - start > 0 else 0.0
    )
    k = len(params)
    aic = likelihood + 2 * k

    return aic, residuals, fitted_values


@njit(cache=True, fastmath=True)
def _difference(y, poly_diff):
    n = len(y)
    m = len(poly_diff)
    if n < m:
        return np.empty(0, dtype=np.float64)
    w = np.empty(n - m + 1, dtype=np.float64)
    for i in range(n - m + 1):
        val = 0.0
        for j in range(m):
            val += poly_diff[j] * y[i + m - 1 - j]
        w[i] = val
    return w


@njit(cache=True, fastmath=True)
def _undifference_sarima(w, initial_values, poly_diff):
    m = len(poly_diff)
    n_init = len(initial_values)
    n_out = n_init + len(w)
    out = np.empty(n_out, dtype=np.float64)
    out[:n_init] = initial_values

    for i in range(len(w)):
        val = w[i]
        for j in range(1, m):
            val -= poly_diff[j] * out[n_init + i - j]
        out[n_init + i] = val
    return out


class SARIMA(BaseForecaster, IterativeForecastingMixin):
    """Seasonal AutoRegressive Integrated Moving Average (SARIMA) forecaster."""

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": True,
    }

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        P: int = 0,
        D: int = 0,
        Q: int = 0,
        seasonal_period: int = 1,
        use_constant: bool = False,
        iterations: int = 200,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.seasonal_period = seasonal_period
        self.use_constant = use_constant
        self.iterations = iterations
        self.c_ = 0.0
        self.phi_ = np.empty(0)
        self.theta_ = np.empty(0)
        self.Phi_ = np.empty(0)
        self.Theta_ = np.empty(0)
        self._series = np.empty(0)
        self._differenced_series = np.empty(0)
        self.residuals_ = np.empty(0)
        self.fitted_values_ = np.empty(0)
        self.aic_ = 0.0
        self._model = np.empty(0)
        self._parameters = np.empty(0)
        self.exog_ = None
        self.beta_ = None
        self.exog_n_features_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        self._series = np.array(y.squeeze(), dtype=np.float64)
        series_for_arima = self._series

        if exog is not None:
            exog = _exog_as_timepoint_rows(exog)
            if len(exog) != len(self._series):
                raise ValueError("exog must have the same number of rows as y")
            self.exog_ = exog
            self.exog_n_features_ = exog.shape[1]
            X = np.column_stack([np.ones(len(self._series)), exog])
            self.beta_ = np.linalg.lstsq(X, self._series, rcond=None)[0]
            series_for_arima = self._series - X @ self.beta_
        else:
            self.beta_ = None
            self.exog_ = None
            self.exog_n_features_ = None

        ar_lags, ma_lags = _get_active_lags(
            self.p, self.q, self.P, self.Q, self.seasonal_period
        )

        model_list = [
            1 if self.use_constant else 0,
            self.p,
            self.q,
            self.P,
            self.Q,
            self.seasonal_period,
            len(ar_lags),
            len(ma_lags),
        ]
        model_list.extend(ar_lags)
        model_list.extend(ma_lags)
        self._model = np.array(model_list, dtype=np.int32)

        poly_diff = _get_diff_poly(self.d, self.D, self.seasonal_period)
        if len(series_for_arima) < len(poly_diff):
            raise ValueError("Series too short for differencing.")
        self._differenced_series = _difference(series_for_arima, poly_diff)

        num_params = (1 if self.use_constant else 0) + self.p + self.q + self.P + self.Q
        s = 0.1 / (num_params + 1)
        self._parameters, self.aic_ = nelder_mead(
            2,
            num_params,
            self._differenced_series,
            self._model,
            max_iter=self.iterations,
            simplex_init=s,
        )

        self.aic_, self.residuals_, self.fitted_values_ = _sarima_model(
            self._parameters,
            self._differenced_series,
            self._model,
        )

        c, phi, theta, Phi, Theta = _extract_sarima_params(
            self._parameters,
            self._model,
        )
        self.c_ = c
        self.phi_ = phi
        self.theta_ = theta
        self.Phi_ = Phi
        self.Theta_ = Theta

        differenced_forecast = self.fitted_values_[-1]
        n_init = len(poly_diff) - 1
        if n_init == 0:
            self.forecast_ = differenced_forecast
        else:
            initial_vals = series_for_arima[-n_init:]
            self.forecast_ = _undifference_sarima(
                np.array([differenced_forecast]),
                initial_vals,
                poly_diff,
            )[n_init]

        return self

    def _predict(self, y, exog=None):
        y = y.squeeze()

        poly_diff = _get_diff_poly(self.d, self.D, self.seasonal_period)
        if len(y) < len(poly_diff):
            raise ValueError("Series too short for differencing.")
        y_diff = _difference(y, poly_diff)

        ar_lags, ma_lags = _get_active_lags(
            self.p, self.q, self.P, self.Q, self.seasonal_period
        )

        poly_ar_ns = np.zeros(self.p + 1)
        poly_ar_ns[0] = 1.0
        if self.p > 0:
            poly_ar_ns[1:] = -self.phi_

        poly_ar_s = np.zeros(self.P * self.seasonal_period + 1)
        poly_ar_s[0] = 1.0
        for j in range(self.P):
            poly_ar_s[(j + 1) * self.seasonal_period] = -self.Phi_[j]

        poly_ar_comb = np.convolve(poly_ar_ns, poly_ar_s)
        ar_values = -poly_ar_comb[ar_lags]

        poly_ma_ns = np.zeros(self.q + 1)
        poly_ma_ns[0] = 1.0
        if self.q > 0:
            poly_ma_ns[1:] = self.theta_

        poly_ma_s = np.zeros(self.Q * self.seasonal_period + 1)
        poly_ma_s[0] = 1.0
        for j in range(self.Q):
            poly_ma_s[(j + 1) * self.seasonal_period] = self.Theta_[j]

        poly_ma_comb = np.convolve(poly_ma_ns, poly_ma_s)
        ma_values = poly_ma_comb[ma_lags]

        n = len(y_diff)
        if n < max(self.p, self.q):
            raise ValueError("Series too short for ARMA(p,q) with given order.")

        residuals = np.zeros(n)
        max_ar_lag = ar_lags[-1] if len(ar_lags) > 0 else 0
        max_ma_lag = ma_lags[-1] if len(ma_lags) > 0 else 0
        start = max(max_ar_lag, max_ma_lag)

        for t in range(start, n):
            ar_term = 0.0
            for j in range(len(ar_lags)):
                ar_term += ar_values[j] * y_diff[t - ar_lags[j]]

            ma_term = 0.0
            for j in range(len(ma_lags)):
                ma_term += ma_values[j] * residuals[t - ma_lags[j]]

            pred = self.c_ + ar_term + ma_term
            residuals[t] = y_diff[t] - pred

        ar_forecast = 0.0
        for j in range(len(ar_lags)):
            ar_forecast += ar_values[j] * y_diff[-ar_lags[j]]

        ma_forecast = 0.0
        for j in range(len(ma_lags)):
            ma_forecast += ma_values[j] * residuals[-ma_lags[j]]

        forecast_diff = self.c_ + ar_forecast + ma_forecast

        reg_component = 0.0
        if self.beta_ is not None:
            if exog is None:
                raise ValueError(
                    "exog must be provided for prediction when model was fit "
                    "with exogenous variables"
                )
            exog_arr = np.asarray(exog)
            if exog_arr.ndim == 1:
                if exog_arr.shape[0] == len(y):
                    exog_row = np.atleast_1d(exog_arr[-1])
                elif exog_arr.shape[0] == 1:
                    exog_row = np.atleast_1d(exog_arr.reshape(1, -1)[0])
                else:
                    exog_row = np.atleast_1d(exog_arr.reshape(-1, 1)[-1])
            else:
                exog_row = exog_arr[-1]
            exog_row = np.asarray(exog_row).reshape(-1)
            if (
                self.exog_n_features_ is not None
                and exog_row.shape[0] != self.exog_n_features_
            ):
                raise ValueError(
                    f"exog must have {self.exog_n_features_} features, "
                    f"got {exog_row.shape[0]}"
                )
            Xf = np.concatenate(([1.0], exog_row), axis=0)
            reg_component = float(Xf @ self.beta_)
        forecast_diff = forecast_diff + reg_component

        n_init = len(poly_diff) - 1
        if n_init == 0:
            return forecast_diff
        else:
            if len(y) >= n_init:
                initial_vals = y[-n_init:]
            else:
                initial_vals = self._series[-n_init:]
            return _undifference_sarima(
                np.array([forecast_diff]),
                initial_vals,
                poly_diff,
            )[n_init]

    def _forecast(self, y, exog=None):
        self._fit(y, exog)
        return float(self.forecast_)

    def _iterative_forecast_from_fitted(self, prediction_horizon, exog=None):
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        self._check_is_fitted()

        h = prediction_horizon
        future_exog = None
        if exog is not None:
            exog = _exog_as_timepoint_rows(exog)
            if exog.shape[0] != h:
                raise ValueError(
                    f"Future exog must have {h} rows (matching prediction_horizon)."
                )
            future_exog = exog
        if self.beta_ is not None and future_exog is not None:
            if future_exog.shape[1] != self.exog_n_features_:
                raise ValueError(
                    f"Future exog must have {self.exog_n_features_} columns."
                )

        ar_lags, ma_lags = _get_active_lags(
            self.p, self.q, self.P, self.Q, self.seasonal_period
        )

        poly_ar_ns = np.zeros(self.p + 1)
        poly_ar_ns[0] = 1.0
        if self.p > 0:
            poly_ar_ns[1:] = -self.phi_

        poly_ar_s = np.zeros(self.P * self.seasonal_period + 1)
        poly_ar_s[0] = 1.0
        for j in range(self.P):
            poly_ar_s[(j + 1) * self.seasonal_period] = -self.Phi_[j]

        poly_ar_comb = np.convolve(poly_ar_ns, poly_ar_s)
        ar_values = -poly_ar_comb[ar_lags]

        poly_ma_ns = np.zeros(self.q + 1)
        poly_ma_ns[0] = 1.0
        if self.q > 0:
            poly_ma_ns[1:] = self.theta_

        poly_ma_s = np.zeros(self.Q * self.seasonal_period + 1)
        poly_ma_s[0] = 1.0
        for j in range(self.Q):
            poly_ma_s[(j + 1) * self.seasonal_period] = self.Theta_[j]

        poly_ma_comb = np.convolve(poly_ma_ns, poly_ma_s)
        ma_values = poly_ma_comb[ma_lags]

        n = len(self._differenced_series)
        residuals = np.zeros(len(self.residuals_) + h)
        residuals[: len(self.residuals_)] = self.residuals_
        forecast_series = np.zeros(n + h)
        forecast_series[:n] = self._differenced_series

        for i in range(h):
            t = n + i
            ar_term = 0.0
            for j in range(len(ar_lags)):
                ar_term += ar_values[j] * forecast_series[t - ar_lags[j]]

            ma_term = 0.0
            for j in range(len(ma_lags)):
                ma_term += ma_values[j] * residuals[t - ma_lags[j]]

            next_value = self.c_ + ar_term + ma_term

            if self.beta_ is not None and future_exog is not None:
                Xf = np.concatenate(([1.0], future_exog[i]))
                next_value += float(Xf @ self.beta_)
            forecast_series[t] = next_value

        y_forecast_diff = forecast_series[n : n + h]
        poly_diff = _get_diff_poly(self.d, self.D, self.seasonal_period)
        n_init = len(poly_diff) - 1
        if n_init == 0:
            return y_forecast_diff
        else:
            return _undifference_sarima(
                y_forecast_diff, self._series[-n_init:], poly_diff
            )[n_init:]
