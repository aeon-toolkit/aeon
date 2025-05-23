"""ARIMAForecaster class.

An implementation of the arima statistics forecasting algorithm.

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = ["ARIMAForecaster"]

from math import comb

import numpy as np

from aeon.forecasting._utils import calc_seasonal_period, kpss_test
from aeon.forecasting.base import BaseForecaster

NOGIL = False
CACHE = True


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecaster.

    An implementation of the Hyndman-Khandakar Auto ARIMA forecasting algorithm[1]_.
    Adjusted to add basic seasonal ARIMA.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.
    """

    def __init__(self, horizon=1):
        super().__init__(horizon=horizon, axis=1)
        self.data_ = []
        self.differenced_data_ = []
        self.residuals_ = []
        self.aic_ = 0
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.ps_ = 0
        self.ds_ = 0
        self.qs_ = 0
        self.seasonal_period_ = 0
        self.constant_term_ = 0
        self.c_ = 0
        self.phi_ = 0
        self.phi_s_ = 0
        self.theta_ = 0
        self.theta_s_ = 0

    def _fit(self, y, exog=None):
        """Fit AutoARIMA forecaster to series y.

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
            Fitted ARIMAForecaster.
        """
        self.data_ = np.array(y.squeeze(), dtype=np.float64)
        (
            self.differenced_data_,
            self.aic_,
            self.p_,
            self.d_,
            self.q_,
            self.ps_,
            self.ds_,
            self.qs_,
            self.seasonal_period_,
            self.constant_term_,
            parameters,
        ) = auto_arima(self.data_)
        (self.c_, self.phi_, self.phi_s_, self.theta_, self.theta_s_) = extract_params(
            parameters, self.p_, self.q_, self.ps_, self.qs_, self.constant_term_
        )
        (
            self.aic_,
            self.residuals_,
        ) = arima_log_likelihood(
            parameters,
            self.differenced_data_,
            self.p_,
            self.q_,
            self.ps_,
            self.qs_,
            self.seasonal_period_,
            self.constant_term_,
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
        y = np.array(y, dtype=np.float64)
        value = calc_arima(
            self.differenced_data_,
            self.p_,
            self.q_,
            self.ps_,
            self.qs_,
            self.seasonal_period_,
            len(self.differenced_data_),
            self.c_,
            self.phi_,
            self.phi_s_,
            self.theta_,
            self.theta_s_,
            self.residuals_,
        )
        history = self.data_[::-1]
        differenced_history = np.diff(self.data_, n=self.d_)[::-1]
        # Step 1: undo seasonal differencing on y^(d)
        for k in range(1, self.ds_ + 1):
            lag = k * self.seasonal_period_
            value += (-1) ** (k + 1) * comb(self.ds_, k) * differenced_history[lag - 1]

        # Step 2: undo ordinary differencing
        for k in range(1, self.d_ + 1):
            value += (-1) ** (k + 1) * comb(self.d_, k) * history[k - 1]

        if y is None:
            return np.array([value])
        else:
            return np.insert(y, 0, value)[:-1]


# Define the ARIMA(p, d, q) likelihood function
def arima_log_likelihood(
    params, data, p, q, ps, qs, seasonal_period, include_constant_term
):
    """Calculate the log-likelihood of an ARIMA model given the parameters."""
    c, phi, phi_s, theta, theta_s = extract_params(
        params, p, q, ps, qs, include_constant_term
    )  # Extract parameters

    # Initialize residuals
    n = len(data)
    residuals = np.zeros(n)
    for t in range(n):
        y_hat = calc_arima(
            data,
            p,
            q,
            ps,
            qs,
            seasonal_period,
            t,
            c,
            phi,
            phi_s,
            theta,
            theta_s,
            residuals,
        )
        residuals[t] = data[t] - y_hat
    # Calculate the log-likelihood
    variance = np.mean(residuals**2)
    liklihood = n * (np.log(2 * np.pi) + np.log(variance) + 1)
    k = len(params)
    aic = liklihood + 2 * k
    return (
        aic,
        residuals,
    )  # Return negative log-likelihood for minimization


def extract_params(params, p, q, ps, qs, include_constant_term):
    """Extract ARIMA parameters from the parameter vector."""
    # Extract parameters
    c = params[0] if include_constant_term else 0  # Constant term
    # AR coefficients
    phi = params[include_constant_term : p + include_constant_term]
    # Seasonal AR coefficients
    phi_s = params[include_constant_term + p : p + ps + include_constant_term]
    # MA coefficients
    theta = params[include_constant_term + p + ps : p + ps + q + include_constant_term]
    # Seasonal MA coefficents
    theta_s = params[
        include_constant_term + p + ps + q : include_constant_term + p + ps + q + qs
    ]
    return c, phi, phi_s, theta, theta_s


def calc_arima(
    data, p, q, ps, qs, seasonal_period, t, c, phi, phi_s, theta, theta_s, residuals
):
    """Calculate the ARIMA forecast for time t."""
    # AR part
    ar_term = 0 if (t - p) < 0 else np.dot(phi, data[t - p : t][::-1])
    # Seasonal AR part
    ars_term = (
        0
        if (t - seasonal_period * ps) < 0
        else np.dot(phi_s, data[t - seasonal_period * ps : t : seasonal_period][::-1])
    )
    # MA part
    ma_term = 0 if (t - q) < 0 else np.dot(theta, residuals[t - q : t][::-1])
    # Seasonal MA part
    mas_term = (
        0
        if (t - seasonal_period * qs) < 0
        else np.dot(
            theta_s, residuals[t - seasonal_period * qs : t : seasonal_period][::-1]
        )
    )
    y_hat = c + ar_term + ma_term + ars_term + mas_term
    return y_hat


def nelder_mead(
    data,
    p,
    q,
    ps,
    qs,
    seasonal_period,
    include_constant_term,
    tol=1e-6,
    max_iter=500,
):
    """Implement the nelder-mead optimisation algorithm."""
    num_params = include_constant_term + p + ps + q + qs
    points = np.full((num_params + 1, num_params), 0.5)
    for i in range(num_params):
        points[i + 1][i] = 0.6
    values = np.array(
        [
            arima_log_likelihood(
                v, data, p, q, ps, qs, seasonal_period, include_constant_term
            )[0]
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
        reflected_value = arima_log_likelihood(
            reflected_point,
            data,
            p,
            q,
            ps,
            qs,
            seasonal_period,
            include_constant_term,
        )[0]
        # if between best and second best, use reflected value
        if len(values) > 1 and values[0] <= reflected_value < values[-2]:
            points[-1] = reflected_point
            values[-1] = reflected_value
            continue
        # Expansion
        # Otherwise if it is better than the best value
        if reflected_value < values[0]:
            expanded_point = centre_point + 2 * (reflected_point - centre_point)
            expanded_value = arima_log_likelihood(
                expanded_point,
                data,
                p,
                q,
                ps,
                qs,
                seasonal_period,
                include_constant_term,
            )[0]
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
        contracted_value = arima_log_likelihood(
            contracted_point,
            data,
            p,
            q,
            ps,
            qs,
            seasonal_period,
            include_constant_term,
        )[0]
        # If contraction is better use that otherwise move to shrinkage
        if contracted_value < values[-1]:
            points[-1] = contracted_point
            values[-1] = contracted_value
            continue

        # Shrinkage
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = arima_log_likelihood(
                points[i],
                data,
                p,
                q,
                ps,
                qs,
                seasonal_period,
                include_constant_term,
            )[0]

        # Convergence check
        if np.max(np.abs(values - values[0])) < tol:
            break
    return points[0], values[0]


# def calc_moving_variance(data, window):
#     X = np.lib.stride_tricks.sliding_window_view(data, window_shape=window)
#     return X.var()


def auto_arima(data):
    """
    Implement the Hyndman-Khandakar algorithm.

    For automatic ARIMA model selection.
    """
    seasonal_period = calc_seasonal_period(data)
    difference = 0
    while not kpss_test(data)[1]:
        data = np.diff(data, n=1)
        difference += 1
    seasonal_difference = 1 if seasonal_period > 1 else 0
    if seasonal_difference:
        data = data[seasonal_period:] - data[:-seasonal_period]
    include_c = 1 if difference == 0 else 0
    model_parameters = [
        [2, 2, 0, 0, include_c],
        [0, 0, 0, 0, include_c],
        [1, 0, 0, 0, include_c],
        [0, 1, 0, 0, include_c],
    ]
    model_points = []
    for p in model_parameters:
        points, aic = nelder_mead(data, p[0], p[1], p[2], p[3], seasonal_period, p[4])
        p.append(aic)
        model_points.append(points)
    current_model = min(model_parameters, key=lambda item: item[5])
    current_points = model_points[model_parameters.index(current_model)]
    while True:
        better_model = False
        for param_no in range(4):
            for adjustment in [-1, 1]:
                if (current_model[param_no] + adjustment) < 0:
                    continue
                model = current_model.copy()
                model[param_no] += adjustment
                for constant_term in [0, 1]:
                    points, aic = nelder_mead(
                        data,
                        model[0],
                        model[1],
                        model[2],
                        model[3],
                        seasonal_period,
                        constant_term,
                    )
                    if aic < current_model[5]:
                        current_model = model
                        current_points = points
                        current_model[5] = aic
                        current_model[4] = constant_term
                        better_model = True
        if not better_model:
            break
    return (
        data,
        current_model[5],
        current_model[0],
        difference,
        current_model[1],
        current_model[2],
        seasonal_difference,
        current_model[3],
        seasonal_period,
        current_model[4],
        current_points,
    )
