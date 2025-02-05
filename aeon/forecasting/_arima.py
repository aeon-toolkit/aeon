"""ARIMAForecaster class.

An implementation of the arima statistics forecasting algorithm.

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = ["ARIMAForecaster"]

import numpy as np

from aeon.forecasting._utils import calc_seasonal_period, kpss_test
from aeon.forecasting.base import BaseForecaster

NOGIL = False
CACHE = True


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecaster.

    An implementation of the exponential smoothing statistics forecasting algorithm.
    Implements additive and multiplicative error models,
    None, additive and multiplicative (including damped) trend and
    None, additive and mutliplicative seasonality[1]_.

    Parameters
    ----------
    alpha : float, default = 0.1
        Level smoothing parameter.
    beta : float, default = 0.01
        Trend smoothing parameter.
    gamma : float, default = 0.01
        Seasonal smoothing parameter.
    phi : float, default = 0.99
        Trend damping smoothing parameters
    horizon : int, default = 1
        The horizon to forecast to.
    model_type : ModelType, default = ModelType()
        A object of type ModelType, describing the error,
        trend and seasonality type of this ETS model.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.

    Examples
    --------
    >>> from aeon.forecasting import ETSForecaster, ModelType
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8, horizon=1,
                               model_type=ModelType(1,2,2,4))
    >>> forecaster.fit(y)
    >>> forecaster.predict()
    366.90200486015596
    """

    def __init__(self, horizon=1):
        super().__init__(horizon=horizon, axis=1)

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
            Fitted BaseForecaster.
        """
        data = np.array(y.squeeze(), dtype=np.float64)
        self.model = auto_arima(data)
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


# Define the ARIMA(p, d, q) likelihood function
def arima_log_likelihood(
    params, data, p, d, q, ps, ds, qs, seasonal_period, include_constant_term
):
    # Extract parameters
    c = params[0]
    phi = params[include_constant_term : p + include_constant_term]  # AR coefficients
    phi_s = params[
        include_constant_term + p : p + ps + include_constant_term
    ]  # Seasonal AR coefficients
    theta = params[
        include_constant_term + p + ps : p + ps + q + include_constant_term
    ]  # MA coefficients
    theta_s = params[include_constant_term + p + ps + q :]  # Seasonal MA coefficents

    # Difference the data, d times
    if d > 0:
        data = np.diff(data, n=d)
    # Seasonal Difference
    for _i in range(ds):
        data = data[seasonal_period:] - data[:-seasonal_period]

    # Initialize residuals
    n = len(data)
    residuals = np.zeros(n)

    for t in range(max(p, q), n):
        # AR part
        ar_term = np.dot(phi, data[t - p : t][::-1])
        # Seasonal AR part
        ars_term = np.dot(
            phi_s, data[t - ps - seasonal_period : t - seasonal_period][::-1]
        )
        # MA part
        ma_term = np.dot(theta, residuals[t - q : t][::-1])
        # Seasonal MA part
        mas_term = np.dot(
            theta_s, residuals[t - qs - seasonal_period : t - seasonal_period][::-1]
        )
        residuals[t] = (
            data[t]
            - (c if include_constant_term else 0)
            - ar_term
            - ma_term
            - ars_term
            - mas_term
        )
    # Calculate the log-likelihood
    variance = (residuals @ residuals) / len(residuals)
    neg_log_likelihood = 0.5 * n * (np.log(2 * np.pi) + np.log(variance) + 1)
    return (
        neg_log_likelihood,
        residuals,
    )  # Return negative log-likelihood for minimization


def nelder_mead(
    data,
    p,
    d,
    q,
    ps,
    ds,
    qs,
    seasonal_period,
    include_constant_term,
    tol=1e-6,
    max_iter=500,
):
    """Implement the nelder-mead optimisation algorithm."""
    points = np.full((1 + p + ps + q + qs + 1, 1 + p + ps + q + qs), 0.5)
    for i in range(1 + p + ps + q + qs):
        points[i + 1][i] = 0.6
    values = np.array(
        [
            arima_log_likelihood(
                v, data, p, d, q, ps, ds, qs, seasonal_period, include_constant_term
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
            d,
            q,
            ps,
            ds,
            qs,
            seasonal_period,
            include_constant_term,
        )[0]
        # if between best and second best, use reflected value
        if values[0] <= reflected_value < values[-2]:
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
                d,
                q,
                ps,
                ds,
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
            d,
            q,
            ps,
            ds,
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
                d,
                q,
                ps,
                ds,
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
    seasonal_period = calc_seasonal_period(data)
    seasonal_difference = seasonal_period > 1
    if seasonal_difference:
        data = data[seasonal_period:] - data[:-seasonal_period]
    difference = 0
    while not kpss_test(data)[1]:
        data = np.diff(data, n=1)
        difference += 1
