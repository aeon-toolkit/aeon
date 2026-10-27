"""Autoregressive forecaster."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ARForecaster"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin
from aeon.forecasting.utils._ar import ar_forecast, ar_ols, burg, yule_walker

_METHODS = {"burg": 0, "yule_walker": 1, "ols": 2}


class ARForecaster(BaseForecaster, IterativeForecastingMixin):
    r"""Autoregressive (AR) forecaster.

    Fits the AR(p) model

    .. math::
        y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t

    and forecasts by running the fitted recursion forward from the end of the
    series. If ``p`` is ``None``, the order is chosen by AIC over
    ``0..max_p``, as in R's ``ar``.

    Three estimators are available:

    - ``"burg"`` (default): Burg's method, which minimises the forward and
      backward prediction errors. The fitted model is always stationary, and
      the estimates are less biased than Yule-Walker on short series or near a
      unit root.
    - ``"yule_walker"``: the Yule-Walker equations on the sample
      autocovariance, solved by the Levinson-Durbin recursion. Also always
      stationary.
    - ``"ols"``: ordinary least squares with an intercept (conditional sum of
      squares). The fitted model is not constrained to be stationary.

    Burg and Yule-Walker give the coefficients of every order in one pass, and
    ``"ols"`` does the same with a single Cholesky factorisation, so selecting
    the order costs little more than fitting one model. With ``"ols"``, all
    orders are scored on the same observations, those after ``max_p``, and the
    selected order is then refitted on all the observations it can use.

    ``predict(y)`` uses the fitted coefficients on the last ``p`` values of the
    series passed to it, with no refit, so it can forecast from new data.

    Parameters
    ----------
    p : int or None, default=None
        AR order. If ``None``, the order is selected by AIC over ``0..max_p``.
    max_p : int or None, default=None
        Largest order considered when ``p`` is ``None``. If ``None``, it is
        ``min(n - 1, floor(10 * log10(n)))`` for a series of length ``n``, as in
        R's ``ar``. It is reduced if the series is too short for it: with
        ``"ols"``, to ``(n - 2) // 2``.
    method : {"burg", "yule_walker", "ols"}, default="burg"
        Estimation method.
    horizon : int, default=1
        Number of steps ahead to forecast. Forecasts more than one step ahead
        run the fitted recursion forward on its own forecasts.

    Attributes
    ----------
    p_ : int
        AR order of the fitted model.
    coef_ : np.ndarray
        Array of shape ``(p_,)``, the coefficients :math:`\phi_1, \dots, \phi_p`.
    intercept_ : float
        The intercept :math:`c`. For ``"burg"`` and ``"yule_walker"`` it is
        :math:`\bar{y}(1 - \sum_j \phi_j)`, where :math:`\bar{y}` is the mean of
        the series.
    sigma2_ : float
        Estimated innovation variance.
    aic_ : float
        AIC of the fitted model, :math:`n_{eff} \log \hat\sigma^2 + 2(p + 1)`,
        where :math:`n_{eff}` is ``n`` for ``"burg"`` and ``"yule_walker"`` and
        ``n - p`` for ``"ols"``.
    forecast_ : float
        Forecast ``horizon`` steps ahead of the series seen in ``fit``.

    See Also
    --------
    ARIMA
    TAR

    References
    ----------
    .. [1] Brockwell, P. J. and Davis, R. A. (2016). Introduction to Time Series
       and Forecasting, 3rd edition, Chapter 5. Springer.
    .. [2] Burg, J. P. (1968). A new analysis technique for time series data.
       NATO Advanced Study Institute on Signal Processing.

    Examples
    --------
    >>> from aeon.forecasting.stats import ARForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ARForecaster(p=2)
    >>> round(forecaster.forecast(y), 2)
    437.85
    """

    _tags = {
        "capability:horizon": True,
    }

    def __init__(self, p=None, max_p=None, method="burg", horizon=1):
        self.p = p
        self.max_p = max_p
        self.method = method
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        y = np.ascontiguousarray(y.ravel(), dtype=np.float64)
        n = y.shape[0]
        if self.method not in _METHODS:
            raise ValueError(
                f"method must be one of {list(_METHODS)}, got {self.method!r}."
            )
        method = _METHODS[self.method]
        if self.p is None:
            max_p = (
                min(n - 1, int(np.floor(10 * np.log10(n))))
                if self.max_p is None
                else int(self.max_p)
            )
            if max_p < 0:
                raise ValueError(f"max_p must be >= 0, got {self.max_p}.")
            limit = (n - 2) // 2 if method == 2 else n - 1
            max_p = max(0, min(max_p, limit))
            p = -1
        else:
            p = int(self.p)
            if p < 0:
                raise ValueError(f"p must be >= 0, got {self.p}.")
            limit = (n - 1) // 2 if method == 2 else n - 1
            if p > limit:
                raise ValueError(
                    f"A series of length {n} is too short to fit an AR({p}) model "
                    f"with method={self.method!r}."
                )
            max_p = p
        order, intercept, coef, sigma2, aic = _fit_ar(y, p, max_p, method)
        self.p_ = int(order)
        self.intercept_ = float(intercept)
        self.coef_ = coef
        self.sigma2_ = float(sigma2)
        self.aic_ = float(aic)
        self.forecast_ = float(ar_forecast(y, intercept, coef, self.horizon)[-1])
        return self

    def _predict(self, y, exog=None):
        y = np.ascontiguousarray(y.ravel(), dtype=np.float64)
        if y.shape[0] < self.p_:
            raise ValueError(
                f"predict needs at least p_={self.p_} values, got {y.shape[0]}."
            )
        return float(ar_forecast(y, self.intercept_, self.coef_, self.horizon)[-1])

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Forecast ``prediction_horizon`` steps ahead with one fit on ``y``.

        Runs the fitted recursion forward in a single call, instead of calling
        ``predict`` once per step. ``ARForecaster`` does not support exogenous
        variables, so ``exog`` and ``future_exog`` are accepted for signature
        compatibility only and raise :class:`NotImplementedError` if provided.

        Parameters
        ----------
        y : np.ndarray
            The time series to fit and forecast from.
        prediction_horizon : int
            The number of future time steps to forecast.
        exog : None
            Not supported.
        future_exog : None
            Not supported.

        Returns
        -------
        np.ndarray
            Array of shape ``(prediction_horizon,)``, the forecasts for steps
            ``1..prediction_horizon``.
        """
        if exog is not None or future_exog is not None:
            raise NotImplementedError("ARForecaster does not support exog.")
        if self.horizon != 1:
            raise ValueError(
                "iterative_forecast requires the forecaster to be configured with "
                f"horizon=1, but ARForecaster has horizon={self.horizon}."
            )
        y, _, _ = self._check_iterative_forecast_inputs(
            y, prediction_horizon, None, None
        )
        self.fit(y)
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        return ar_forecast(y, self.intercept_, self.coef_, prediction_horizon)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return [
            {},
            {"p": 2, "method": "ols"},
            {"max_p": 3, "method": "yule_walker", "horizon": 2},
        ]


@njit(cache=True)
def _aic(n_eff, sigma2, k):
    if sigma2 < 1e-300:
        sigma2 = 1e-300
    return n_eff * np.log(sigma2) + 2.0 * (k + 1)


@njit(cache=True)
def _fit_ar(y, p, max_p, method):
    """Fit an AR model of order ``p``, or select the order by AIC if ``p < 0``.

    ``method`` is 0 for Burg, 1 for Yule-Walker and 2 for least squares.
    Returns ``(order, intercept, coef, sigma2, aic)``.
    """
    n = y.shape[0]
    if method == 2:
        coefs, intercepts, rss = ar_ols(y, max_p, max_p)
        rows = n - max_p
        order = max_p
        if p < 0:
            best = np.inf
            for k in range(max_p + 1):
                a = _aic(rows, rss[k] / rows, k)
                if a < best:
                    best = a
                    order = k
            if order < max_p:
                # Refit the selected order on all the observations it can use.
                coefs, intercepts, rss = ar_ols(y, order, order)
                rows = n - order
        sigma2 = rss[order] / rows
        return (
            order,
            intercepts[order],
            coefs[order, :order].copy(),
            sigma2,
            _aic(rows, sigma2, order),
        )
    if method == 0:
        coefs, sigma2s, _ = burg(y, max_p)
    else:
        coefs, sigma2s, _ = yule_walker(y, max_p)
    order = max_p
    if p < 0:
        best = np.inf
        for k in range(max_p + 1):
            a = _aic(n, sigma2s[k], k)
            if a < best:
                best = a
                order = k
    coef = coefs[order, :order].copy()
    intercept = np.mean(y) * (1.0 - np.sum(coef))
    return order, intercept, coef, sigma2s[order], _aic(n, sigma2s[order], order)
