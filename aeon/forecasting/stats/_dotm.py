"""Dynamic Optimised Theta Model (DOTM) forecaster."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["DOTM"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin

# 95% one-sided normal quantile used by the ACF seasonal test (StatsForecast).
_ACF_TEST_QUANTILE = 1.6448536269514722

# Floor for multiplicative seasonal factors; below this we fall back to additive
# decomposition to avoid blowing up the deseasonalised series.
_MULTIPLICATIVE_FLOOR = 1e-8


class DOTM(BaseForecaster, IterativeForecastingMixin):
    """Dynamic Optimised Theta Model (DOTM) forecaster.

    Implementation of the Dynamic Optimised Theta Model (DOTM) proposed by
    Fiorucci et al. (2016) [1]_. DOTM is a local univariate forecaster that
    estimates an initial level, smoothing parameter, and theta parameter
    unless they are fixed by the user.

    Seasonality is handled as an outer transformation: when seasonal
    adjustment is requested or detected, the input series is decomposed into
    a seasonal component and a seasonally adjusted component using classical
    (additive or multiplicative) decomposition with a centred moving-average
    trend estimate. The DOTM core is then fitted to the adjusted series, and
    forecasts are produced by recombining the DOTM forecast with a seasonal
    naive forecast of the seasonal component. The default behaviour is
    non-seasonal because ``season_length=1``.

    Exogenous variables and prediction intervals are not supported.

    Parameters
    ----------
    initial_level : float or None, default=None
        Fixed initial level, ``ell0``. If ``None``, it is estimated.
    alpha : float or None, default=None
        Fixed smoothing parameter. If ``None``, it is estimated.
    theta : float or None, default=None
        Fixed theta parameter. If ``None``, it is estimated.
    initial_level_bounds : tuple of float, default=(-1e10, 1e10)
        Bounds for estimated ``initial_level``.
    alpha_bounds : tuple of float, default=(0.1, 0.99)
        Bounds for estimated ``alpha``.
    theta_bounds : tuple of float, default=(1.0, 1e10)
        Bounds for estimated ``theta``.
    season_length : int, default=1
        Seasonal period. ``1`` means no seasonality.
    decomposition_type : {"multiplicative", "additive"}, default="multiplicative"
        Type of classical decomposition used when the series is deseasonalised.
        Multiplicative decomposition falls back to additive when the input
        contains non-positive values or produces seasonal factors that are
        non-finite or smaller than ``1e-8``.
    seasonal_test : {"auto", True, False}, default="auto"
        Controls whether to deseasonalise the series.

        - ``False`` : never deseasonalise.
        - ``True``  : deseasonalise when ``season_length > 1`` and at least
          two full seasonal cycles are available.
        - ``"auto"``: apply an ACF-based seasonal test at lag ``season_length``
          to the first-differenced series and deseasonalise only when seasonal
          evidence is present. First differencing removes a constant trend, so
          monotone series do not trigger spurious seasonal detection.
    max_iter : int, default=500
        Maximum number of Nelder-Mead iterations.
    tol : float, default=1e-6
        Simplex convergence tolerance.

    Attributes
    ----------
    initial_level_ : float
        Fitted initial level (from the adjusted-series fit).
    alpha_ : float
        Fitted smoothing parameter.
    theta_ : float
        Fitted theta parameter.
    fitted_values_ : np.ndarray
        In-sample one-step-ahead fitted values on the original scale.
    residuals_ : np.ndarray
        In-sample residuals on the original scale (``y - fitted_values_``).
    forecast_ : float
        Stored one-step-ahead forecast on the original scale.
    sse_ : float
        Scaled in-sample SSE objective value from the DOTM core fit on the
        adjusted series.
    level_ : float
        Final level state after fitting (adjusted series).
    a_ : float
        Final dynamic line intercept state.
    b_ : float
        Final dynamic line slope state.
    mean_y_ : float
        Final running mean state.
    season_length_ : int
        Seasonal period actually used. ``1`` when no deseasonalisation was
        performed.
    decomposition_type_ : str
        Decomposition type actually used. May differ from
        ``decomposition_type`` if multiplicative fell back to additive.
    deseasonalised_ : bool
        Whether the input series was deseasonalised before fitting.
    seasonal_factors_ : np.ndarray or None
        Estimated seasonal factors of length ``season_length_``, or ``None``
        when ``deseasonalised_`` is ``False``.

    References
    ----------
    .. [1] Fiorucci, J. A., Pellegrini, T. R., Louzada, F.,
       Petropoulos, F., & Koehler, A. B. (2016). Models for optimising
       the theta method and their relationship to state space models.
       International Journal of Forecasting, 32(4), 1151-1161.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting.stats import DOTM
    >>> y = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1])
    >>> forecaster = DOTM()
    >>> pred = forecaster.iterative_forecast(y, prediction_horizon=2)
    >>> pred.shape
    (2,)
    """

    _tags = {"capability:horizon": False}

    def __init__(
        self,
        initial_level=None,
        alpha=None,
        theta=None,
        initial_level_bounds=(-1e10, 1e10),
        alpha_bounds=(0.1, 0.99),
        theta_bounds=(1.0, 1e10),
        season_length=1,
        decomposition_type="multiplicative",
        seasonal_test="auto",
        max_iter=500,
        tol=1e-6,
    ):
        self.initial_level = initial_level
        self.alpha = alpha
        self.theta = theta
        self.initial_level_bounds = initial_level_bounds
        self.alpha_bounds = alpha_bounds
        self.theta_bounds = theta_bounds
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.seasonal_test = seasonal_test
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit DOTM to a univariate series, optionally with seasonal adjustment."""
        self._validate_seasonal_params()
        y = _prepare_dotm_y(y)
        y_original = y

        deseasonalise = _should_deseasonalise(y, self.season_length, self.seasonal_test)

        if deseasonalise:
            factors, y_adjusted, used_type = _seasonal_decompose(
                y, self.season_length, self.decomposition_type
            )
            # If seasonality requested by could not be performed used_type is none.
            # n is too short for a centred MA of period m
            # one or more  positions ended up with no finite samples after MA-trim
            # the normalised factors or the adjusted series came out non-finite.
            if used_type == "none":
                deseasonalise = False
                y_adjusted = y
                self.season_length_ = 1
                self.decomposition_type_ = self.decomposition_type
                self.seasonal_factors_ = None
            else:
                self.season_length_ = int(self.season_length)
                self.decomposition_type_ = used_type
                self.seasonal_factors_ = factors
        else:
            y_adjusted = y
            self.season_length_ = 1
            self.decomposition_type_ = self.decomposition_type
            self.seasonal_factors_ = None

        self.deseasonalised_ = deseasonalise
        self._y_adjusted_ = y_adjusted

        fixed_mask, fixed_values, lower, upper = self._parameter_arrays()
        x0 = np.array([y_adjusted[0] / 2.0, 0.5, 2.0], dtype=np.float64)
        for i in range(3):
            if fixed_mask[i]:
                x0[i] = fixed_values[i]
            elif x0[i] < lower[i]:
                x0[i] = lower[i]
            elif x0[i] > upper[i]:
                x0[i] = upper[i]

        (
            params,
            self.sse_,
            adjusted_fitted,
            adjusted_residuals,
            level,
            a,
            b,
            mean_y,
            adjusted_forecast,
        ) = _fit_dotm_core(
            x0,
            y_adjusted,
            fixed_mask,
            fixed_values,
            lower,
            upper,
            float(self.tol),
            int(self.max_iter),
        )

        self.initial_level_ = float(params[0])
        self.alpha_ = float(params[1])
        self.theta_ = float(params[2])
        self.level_ = float(level)
        self.a_ = float(a)
        self.b_ = float(b)
        self.mean_y_ = float(mean_y)

        if self.deseasonalised_:
            n = y_original.shape[0]
            in_sample = self.seasonal_factors_[np.arange(n) % self.season_length_]
            one_step_factor = self.seasonal_factors_[n % self.season_length_]
            if self.decomposition_type_ == "additive":
                self.fitted_values_ = adjusted_fitted + in_sample
                self.forecast_ = float(adjusted_forecast + one_step_factor)
            else:
                self.fitted_values_ = adjusted_fitted * in_sample
                self.forecast_ = float(adjusted_forecast * one_step_factor)
            self.residuals_ = y_original - self.fitted_values_
        else:
            self.fitted_values_ = adjusted_fitted
            self.residuals_ = adjusted_residuals
            self.forecast_ = float(adjusted_forecast)

        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead from ``y`` using fitted parameters.

        For seasonal models the supplied context is deseasonalised using the
        *fitted* seasonal factors (applied by position), the DOTM core is run
        on the adjusted context, and the result is re-seasonalised using the
        factor at position ``len(y) % season_length_``. This makes rolling
        one-step ``predict`` consistent with multi-step
        ``iterative_forecast``, at the cost of assuming the supplied context
        shares its seasonal phase with the training data.
        """
        y = _prepare_dotm_y(y)
        if not self.deseasonalised_:
            return float(
                _dotm_forecast(y, 1, self.initial_level_, self.alpha_, self.theta_)[0]
            )

        n = y.shape[0]
        m = self.season_length_
        factors = self.seasonal_factors_
        rep = factors[np.arange(n) % m]
        if self.decomposition_type_ == "additive":
            y_adjusted = y - rep
        else:
            y_adjusted = y / rep
        adjusted_fc = _dotm_forecast(
            y_adjusted, 1, self.initial_level_, self.alpha_, self.theta_
        )[0]
        next_factor = factors[n % m]
        if self.decomposition_type_ == "additive":
            return float(adjusted_fc + next_factor)
        return float(adjusted_fc * next_factor)

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        """Fit DOTM on ``y`` and recursively forecast ``prediction_horizon`` steps."""
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        if exog is not None:
            raise NotImplementedError("DOTM does not support exog.")
        self.fit(y)
        h = int(prediction_horizon)

        # The recurrence state at position n - 1 is already stored after fit;
        # extend forward from there rather than re-walking the in-sample series.
        n = self._y_adjusted_.shape[0]
        adjusted_fc = _dotm_forecast_from_state(
            n,
            h,
            self.level_,
            self.a_,
            self.b_,
            self.mean_y_,
            self.alpha_,
            self.theta_,
        )
        if not self.deseasonalised_:
            return adjusted_fc

        seasonal_fc = _seasonal_forecast(
            self.seasonal_factors_, h, self.season_length_, n
        )
        if self.decomposition_type_ == "additive":
            return adjusted_fc + seasonal_fc
        return adjusted_fc * seasonal_fc

    def _validate_seasonal_params(self):
        """Validate seasonal-extension parameters in ``_fit`` (not ``__init__``)."""
        if isinstance(self.season_length, bool) or not isinstance(
            self.season_length, (int, np.integer)
        ):
            raise ValueError("season_length must be a positive integer.")
        if self.season_length < 1:
            raise ValueError("season_length must be at least 1.")
        if self.decomposition_type not in ("multiplicative", "additive"):
            raise ValueError(
                "decomposition_type must be 'multiplicative' or 'additive', "
                f"got {self.decomposition_type!r}."
            )
        # Accept the three allowed values without confusing True with 1.
        if not (
            self.seasonal_test == "auto"
            or self.seasonal_test is True
            or self.seasonal_test is False
        ):
            raise ValueError(
                "seasonal_test must be 'auto', True, or False, "
                f"got {self.seasonal_test!r}."
            )

    def _parameter_arrays(self):
        """Return fixed-parameter masks, values, and bounds."""
        fixed_values = np.zeros(3, dtype=np.float64)
        fixed_mask = np.zeros(3, dtype=np.bool_)
        lower = np.array(
            [
                self.initial_level_bounds[0],
                self.alpha_bounds[0],
                self.theta_bounds[0],
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                self.initial_level_bounds[1],
                self.alpha_bounds[1],
                self.theta_bounds[1],
            ],
            dtype=np.float64,
        )
        fixed_params = (self.initial_level, self.alpha, self.theta)
        for i, value in enumerate(fixed_params):
            if value is not None:
                fixed_values[i] = float(value)
                fixed_mask[i] = True
            else:
                fixed_values[i] = np.nan

        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("DOTM parameter bounds must be finite.")
        if np.any(lower > upper):
            raise ValueError("DOTM lower bounds must not exceed upper bounds.")
        for i in range(3):
            if fixed_mask[i] and (
                not np.isfinite(fixed_values[i])
                or fixed_values[i] < lower[i]
                or fixed_values[i] > upper[i]
            ):
                raise ValueError("Fixed DOTM parameters must be finite and in bounds.")
        if lower[1] <= 0.0 or upper[1] >= 1.0:
            raise ValueError("alpha bounds must lie inside (0, 1).")
        if lower[2] < 1.0:
            raise ValueError("theta lower bound must be at least 1.0.")
        return fixed_mask, fixed_values, lower, upper


# ---------------------------------------------------------------------------
# Seasonal helpers
# ---------------------------------------------------------------------------


def _prepare_dotm_y(y):
    """Convert DOTM input to a 1D array and check minimum length."""
    y = np.asarray(y, dtype=np.float64).squeeze()
    if y.shape[0] < 4:
        raise ValueError("DOTM requires at least 4 observations.")
    return y


def _should_deseasonalise(y, season_length, seasonal_test):
    """Decide whether the series should be seasonally adjusted before fitting.

    The ``"auto"`` mode delegates the ACF significance test to the numba
    implementation :func:`_acf_seasonal_test_numba`.
    """
    if season_length <= 1:
        return False
    if y.shape[0] < 2 * season_length:
        return False
    if seasonal_test is True:
        return True
    if seasonal_test is False:
        return False
    return bool(_acf_seasonal_test_numba(y, int(season_length), _ACF_TEST_QUANTILE))


def _seasonal_decompose(y, season_length, requested_type):
    """Thin wrapper around :func:`_seasonal_decompose_numba`.

    Translates the integer type code returned from numba into the public
    ``"multiplicative"`` / ``"additive"`` / ``"none"`` string and unifies the
    ``None`` factor return when no decomposition was applied.
    """
    type_code, factors, adjusted = _seasonal_decompose_numba(
        y,
        int(season_length),
        requested_type == "multiplicative",
        _MULTIPLICATIVE_FLOOR,
    )
    if type_code == 0:
        return None, adjusted, "none"
    if type_code == 1:
        return factors, adjusted, "additive"
    return factors, adjusted, "multiplicative"


def _seasonal_forecast(seasonal_factors, h, season_length, n_train):
    """Repeat the learned seasonal cycle in phase for ``h`` future steps."""
    return seasonal_factors[(int(n_train) + np.arange(int(h))) % int(season_length)]


@njit(cache=True, fastmath=True)
def _acf_seasonal_test_numba(y, season_length, threshold):
    """Single-pass ACF seasonal significance test on first-differenced ``y``.

    Equivalent to the previous numpy implementation: compute ``dy = diff(y)``,
    centre it, then compare the Bartlett-corrected ACF at lag ``season_length``
    against ``threshold``. Implemented as one numba routine to avoid the
    per-call numpy roundtrip overhead seen on short series.
    """
    n = y.shape[0]
    if n <= season_length + 1:
        return False

    dy_len = n - 1
    mean_dy = 0.0
    for i in range(dy_len):
        mean_dy += y[i + 1] - y[i]
    mean_dy /= dy_len

    dy_centred = np.empty(dy_len, dtype=np.float64)
    denom = 0.0
    for i in range(dy_len):
        c = (y[i + 1] - y[i]) - mean_dy
        dy_centred[i] = c
        denom += c * c
    if denom <= 1e-12:
        return False

    m = season_length
    prior_sq = 0.0
    acf_at_m = 0.0
    for lag in range(1, m + 1):
        s = 0.0
        for i in range(dy_len - lag):
            s += dy_centred[i] * dy_centred[i + lag]
        acf = s / denom
        if lag < m:
            prior_sq += acf * acf
        else:
            acf_at_m = acf

    se_sq = (1.0 + 2.0 * prior_sq) / dy_len
    if se_sq <= 0.0 or not np.isfinite(se_sq):
        return False
    se = se_sq**0.5
    return abs(acf_at_m) / se > threshold


@njit(cache=True, fastmath=True)
def _seasonal_decompose_numba(y, season_length, requested_mult, mult_floor):
    """Classical seasonal decomposition with centred-MA detrending in one pass.

    Steps mirror the previous numpy version:

    1. Estimate a trend via the centred moving average of period
       ``season_length`` (odd ``m``: uniform window; even ``m``: 2*m smoothing
       with ``[0.5, 1, ..., 1, 0.5] / m`` weights).
    2. Detrend additively or multiplicatively.
    3. Estimate seasonal indices as the per-position mean of the detrended
       series, ignoring positions where the MA is undefined.
    4. Normalise (sum-zero for additive, mean-one for multiplicative).

    Multiplicative falls back to additive when ``y`` contains non-positive
    values, when the MA contains non-positive values, or when factors are
    non-finite or below ``mult_floor``.

    Returns
    -------
    type_code : int
        ``0`` for "none" (no decomposition applied), ``1`` for additive,
        ``2`` for multiplicative.
    factors : np.ndarray
        Seasonal factors of length ``season_length``. All zeros when
        ``type_code == 0``; the caller should ignore the array in that case.
    adjusted : np.ndarray
        Seasonally adjusted series of the same length as ``y``. Equal to a
        copy of ``y`` when ``type_code == 0``.
    """
    n = y.shape[0]
    m = season_length
    half = m // 2

    factors = np.zeros(m, dtype=np.float64)

    # Centred MA requires at least m points (odd m) or m + 1 points (even m).
    if m % 2 == 1:
        if n < m:
            return 0, factors, y.copy()
    else:
        if n < m + 1:
            return 0, factors, y.copy()

    inv_m = 1.0 / m
    ma = np.empty(n, dtype=np.float64)
    for i in range(n):
        ma[i] = np.nan

    if m % 2 == 1:
        for t in range(half, n - half):
            s = 0.0
            for k in range(t - half, t + half + 1):
                s += y[k]
            ma[t] = s * inv_m
    else:
        for t in range(half, n - half):
            s = 0.5 * (y[t - half] + y[t + half])
            for k in range(t - half + 1, t + half):
                s += y[k]
            ma[t] = s * inv_m

    # Multiplicative viability check.
    use_mult = requested_mult
    if use_mult:
        for i in range(n):
            if y[i] <= 0.0:
                use_mult = False
                break
    if use_mult:
        for t in range(half, n - half):
            if ma[t] <= 0.0:
                use_mult = False
                break

    sums = np.empty(m, dtype=np.float64)
    counts = np.empty(m, dtype=np.int64)

    if use_mult:
        for p in range(m):
            sums[p] = 0.0
            counts[p] = 0
        for t in range(half, n - half):
            d = y[t] / ma[t]
            if np.isfinite(d):
                pos = t % m
                sums[pos] += d
                counts[pos] += 1

        valid = True
        for p in range(m):
            if counts[p] == 0:
                valid = False
                break
            factors[p] = sums[p] / counts[p]

        if valid:
            factor_sum = 0.0
            for p in range(m):
                factor_sum += factors[p]
            factor_mean = factor_sum / m
            if np.isfinite(factor_mean) and factor_mean > mult_floor:
                ok = True
                min_factor = 1e300
                for p in range(m):
                    f = factors[p] / factor_mean
                    factors[p] = f
                    if not np.isfinite(f):
                        ok = False
                        break
                    if f < min_factor:
                        min_factor = f
                if ok and min_factor >= mult_floor:
                    adjusted = np.empty(n, dtype=np.float64)
                    all_finite = True
                    for i in range(n):
                        rep = factors[i % m]
                        v = y[i] / rep
                        adjusted[i] = v
                        if not np.isfinite(v):
                            all_finite = False
                            break
                    if all_finite:
                        return 2, factors, adjusted
        # Fall through to additive on any multiplicative failure.

    # Additive path (also the multiplicative fallback).
    for p in range(m):
        sums[p] = 0.0
        counts[p] = 0
    for t in range(half, n - half):
        d = y[t] - ma[t]
        if np.isfinite(d):
            pos = t % m
            sums[pos] += d
            counts[pos] += 1

    for p in range(m):
        if counts[p] == 0:
            zeros = np.zeros(m, dtype=np.float64)
            return 0, zeros, y.copy()
        factors[p] = sums[p] / counts[p]

    factor_sum = 0.0
    for p in range(m):
        factor_sum += factors[p]
    factor_mean = factor_sum / m
    for p in range(m):
        factors[p] -= factor_mean
        if not np.isfinite(factors[p]):
            zeros = np.zeros(m, dtype=np.float64)
            return 0, zeros, y.copy()

    adjusted = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = y[i] - factors[i % m]
        adjusted[i] = v
        if not np.isfinite(v):
            zeros = np.zeros(m, dtype=np.float64)
            return 0, zeros, y.copy()
    return 1, factors, adjusted


# ---------------------------------------------------------------------------
# Numba core (unchanged from the non-seasonal implementation)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _fit_dotm_core(x0, y, fixed_mask, fixed_values, lower, upper, tol, max_iter):
    """Fit DOTM and compute final fitted-state outputs in one numba call."""
    params, sse = _bounded_nelder_mead_dotm(
        x0,
        y,
        fixed_mask,
        fixed_values,
        lower,
        upper,
        tol,
        max_iter,
    )
    fitted, residuals, level, a, b, mean_y = _dotm_fitted_values(
        y,
        params[0],
        params[1],
        params[2],
    )
    # Reuse the final state from _dotm_fitted_values to produce the 1-step
    # forecast in O(1) rather than re-iterating through y from index 0.
    forecast = _dotm_forecast_from_state(
        y.shape[0], 1, level, a, b, mean_y, params[1], params[2]
    )[0]
    return params, sse, fitted, residuals, level, a, b, mean_y, forecast


@njit(cache=True, fastmath=True)
def _dotm_forecast_from_state(
    n, h, level, a_state, b_state, mean_y_state, alpha, theta
):
    """Project ``h`` DOTM forecasts forward from the state at position ``n - 1``.

    ``level``, ``a_state``, ``b_state`` and ``mean_y_state`` are the values
    stored by :func:`_dotm_fitted_values` after consuming ``y[0..n-1]``.
    Forecasting from the saved state is O(h) and avoids re-walking the
    in-sample series, which is the dominant cost of repeated fit-then-forecast
    use such as :meth:`DOTM.iterative_forecast`.
    """
    forecast = np.empty(h, dtype=np.float64)
    omega = 1.0 - 1.0 / theta
    one_minus_alpha = 1.0 - alpha

    ell = level
    a = a_state
    b = b_state
    mean_y = mean_y_state

    # The first forecast is computed at i = n - 1, so power = (1 - alpha) ** n.
    # We then update power = power * (1 - alpha) each iteration to avoid the
    # ``**`` call that the original recurrence performed inside the loop.
    power = one_minus_alpha**n

    for k in range(h):
        i = n - 1 + k
        next_power = power * one_minus_alpha
        if alpha > 0.0:
            slope_term = b * (1.0 - next_power) / alpha
        else:
            slope_term = 0.0
        mu = ell + omega * (a * power + slope_term)
        forecast[k] = mu

        # Update state to position i + 1, mirroring _dotm_fitted_values but
        # with ``next_y = mu`` since we are extending past the end of y.
        ell = alpha * mu + one_minus_alpha * ell
        previous_mean = mean_y
        mean_y = ((i + 1) * mean_y + mu) / (i + 2)
        b = (i * b + 6.0 * (mu - previous_mean) / (i + 2)) / (i + 3)
        a = mean_y - b * (i + 3) / 2.0

        power = next_power

    return forecast


@njit(cache=True, fastmath=True)
def _dotm_fitted_values(y, initial_level, alpha, theta):
    """Compute DOTM in-sample fitted values and final states."""
    n = y.shape[0]
    fitted = np.empty(n, dtype=np.float64)
    residuals = np.empty(n, dtype=np.float64)
    ell = np.empty(n, dtype=np.float64)
    mean_y = np.empty(n, dtype=np.float64)
    a = np.empty(n, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    omega = 1.0 - 1.0 / theta

    ell[0] = alpha * y[0] + (1.0 - alpha) * initial_level
    mean_y[0] = y[0]
    a[0] = y[0]
    b[0] = 0.0
    fitted[0] = y[0]
    residuals[0] = y[0] - fitted[0]

    for i in range(0, n - 1):
        power = (1.0 - alpha) ** (i + 1)
        if alpha <= 0.0:
            slope_term = 0.0
        else:
            slope_term = b[i] * (1.0 - (1.0 - alpha) ** (i + 2)) / alpha
        fitted[i + 1] = ell[i] + omega * (a[i] * power + slope_term)
        residuals[i + 1] = y[i + 1] - fitted[i + 1]
        ell[i + 1] = alpha * y[i + 1] + (1.0 - alpha) * ell[i]
        mean_y[i + 1] = ((i + 1) * mean_y[i] + y[i + 1]) / (i + 2)
        b[i + 1] = (i * b[i] + 6.0 * (y[i + 1] - mean_y[i]) / (i + 2)) / (i + 3)
        a[i + 1] = mean_y[i + 1] - b[i + 1] * (i + 3) / 2.0

    return fitted, residuals, ell[-1], a[-1], b[-1], mean_y[-1]


@njit(cache=True, fastmath=True)
def _dotm_forecast(y, h, initial_level, alpha, theta):
    """Recursively compute DOTM forecasts beyond the end of ``y``."""
    n = y.shape[0]
    forecast = np.empty(h, dtype=np.float64)
    omega = 1.0 - 1.0 / theta

    ell = alpha * y[0] + (1.0 - alpha) * initial_level
    mean_y = y[0]
    a = y[0]
    b = 0.0

    for i in range(0, n + h - 1):
        power = (1.0 - alpha) ** (i + 1)
        slope_term = b * (1.0 - (1.0 - alpha) ** (i + 2)) / alpha
        mu = ell + omega * (a * power + slope_term)
        if i >= n - 1:
            next_y = mu
            forecast[i - n + 1] = mu
        else:
            next_y = y[i + 1]
        ell = alpha * next_y + (1.0 - alpha) * ell
        previous_mean = mean_y
        mean_y = ((i + 1) * mean_y + next_y) / (i + 2)
        b = (i * b + 6.0 * (next_y - previous_mean) / (i + 2)) / (i + 3)
        a = mean_y - b * (i + 3) / 2.0

    return forecast


@njit(cache=True, fastmath=True)
def _dotm_sse(params, y, fixed_mask, fixed_values, lower, upper):
    """Scaled DOTM SSE objective with bound penalties."""
    initial_level = fixed_values[0] if fixed_mask[0] else params[0]
    alpha = fixed_values[1] if fixed_mask[1] else params[1]
    theta = fixed_values[2] if fixed_mask[2] else params[2]
    if (
        not np.isfinite(initial_level)
        or initial_level < lower[0]
        or initial_level > upper[0]
    ):
        return 1e300
    if not np.isfinite(alpha) or alpha < lower[1] or alpha > upper[1]:
        return 1e300
    if not np.isfinite(theta) or theta < lower[2] or theta > upper[2]:
        return 1e300
    if alpha <= 0.0 or alpha >= 1.0 or theta < 1.0:
        return 1e300

    return _dotm_sse_values(initial_level, alpha, theta, y)


@njit(cache=True, fastmath=True)
def _dotm_sse_values(initial_level, alpha, theta, y):
    """Scaled DOTM SSE objective for concrete parameter values."""
    omega = 1.0 - 1.0 / theta

    scale = 0.0
    for i in range(y.shape[0]):
        scale += abs(y[i])
    scale /= y.shape[0]
    if scale <= 1e-12 or not np.isfinite(scale):
        scale = 1.0

    ell = alpha * y[0] + (1.0 - alpha) * initial_level
    mean_y = y[0]
    a = y[0]
    b = 0.0
    sse = 0.0
    for i in range(0, y.shape[0] - 1):
        power = (1.0 - alpha) ** (i + 1)
        slope_term = b * (1.0 - (1.0 - alpha) ** (i + 2)) / alpha
        fitted = ell + omega * (a * power + slope_term)
        residual = y[i + 1] - fitted
        if not np.isfinite(fitted) or not np.isfinite(residual):
            return 1e300
        if i + 1 >= 2:
            err = residual / scale
            sse += err * err
        ell = alpha * y[i + 1] + (1.0 - alpha) * ell
        previous_mean = mean_y
        mean_y = ((i + 1) * mean_y + y[i + 1]) / (i + 2)
        b = (i * b + 6.0 * (y[i + 1] - previous_mean) / (i + 2)) / (i + 3)
        a = mean_y - b * (i + 3) / 2.0
    if not np.isfinite(sse):
        return 1e300
    return sse


@njit(cache=True, fastmath=True)
def _bounded_nelder_mead_dotm(
    x0, y, fixed_mask, fixed_values, lower, upper, tol, max_iter
):
    """Small bounded Nelder-Mead optimiser for DOTM parameters."""
    free_count = 0
    for i in range(3):
        if not fixed_mask[i]:
            free_count += 1

    full_best = np.empty(3, dtype=np.float64)
    for i in range(3):
        full_best[i] = fixed_values[i] if fixed_mask[i] else x0[i]

    if free_count == 0:
        best = _dotm_sse(full_best, y, fixed_mask, fixed_values, lower, upper)
        return full_best, best

    free_idx = np.empty(free_count, dtype=np.int64)
    pos = 0
    for i in range(3):
        if not fixed_mask[i]:
            free_idx[pos] = i
            pos += 1

    simplex = np.empty((free_count + 1, free_count), dtype=np.float64)
    scores = np.empty(free_count + 1, dtype=np.float64)
    for j in range(free_count):
        simplex[0, j] = x0[free_idx[j]]
    for i in range(1, free_count + 1):
        for j in range(free_count):
            simplex[i, j] = simplex[0, j]
        idx = free_idx[i - 1]
        step = 0.05 * abs(simplex[0, i - 1])
        if step <= 1e-6:
            step = 0.05 * (upper[idx] - lower[idx])
        if step <= 1e-6:
            step = 1e-3
        simplex[i, i - 1] += step

    for i in range(free_count + 1):
        scores[i] = _free_score(
            simplex[i], y, fixed_mask, fixed_values, lower, upper, free_idx
        )

    centroid = np.empty(free_count, dtype=np.float64)
    reflected = np.empty(free_count, dtype=np.float64)
    expanded = np.empty(free_count, dtype=np.float64)
    contracted = np.empty(free_count, dtype=np.float64)
    for _ in range(max_iter):
        _sort_simplex(simplex, scores)
        if _simplex_converged(simplex, scores, tol):
            break

        for j in range(free_count):
            centroid[j] = 0.0
        for i in range(free_count):
            for j in range(free_count):
                centroid[j] += simplex[i, j]
        for j in range(free_count):
            centroid[j] /= free_count

        for j in range(free_count):
            reflected[j] = centroid[j] + (centroid[j] - simplex[free_count, j])
        reflected_score = _free_score(
            reflected, y, fixed_mask, fixed_values, lower, upper, free_idx
        )
        if reflected_score < scores[0]:
            for j in range(free_count):
                expanded[j] = centroid[j] + 2.0 * (reflected[j] - centroid[j])
            expanded_score = _free_score(
                expanded, y, fixed_mask, fixed_values, lower, upper, free_idx
            )
            if expanded_score < reflected_score:
                for j in range(free_count):
                    simplex[free_count, j] = expanded[j]
                scores[free_count] = expanded_score
            else:
                for j in range(free_count):
                    simplex[free_count, j] = reflected[j]
                scores[free_count] = reflected_score
        elif reflected_score < scores[free_count - 1]:
            for j in range(free_count):
                simplex[free_count, j] = reflected[j]
            scores[free_count] = reflected_score
        else:
            if reflected_score < scores[free_count]:
                for j in range(free_count):
                    contracted[j] = centroid[j] + 0.5 * (reflected[j] - centroid[j])
            else:
                for j in range(free_count):
                    contracted[j] = centroid[j] + 0.5 * (
                        simplex[free_count, j] - centroid[j]
                    )
            contracted_score = _free_score(
                contracted, y, fixed_mask, fixed_values, lower, upper, free_idx
            )
            if contracted_score < scores[free_count]:
                for j in range(free_count):
                    simplex[free_count, j] = contracted[j]
                scores[free_count] = contracted_score
            else:
                for i in range(1, free_count + 1):
                    for j in range(free_count):
                        simplex[i, j] = simplex[0, j] + 0.5 * (
                            simplex[i, j] - simplex[0, j]
                        )
                    scores[i] = _free_score(
                        simplex[i],
                        y,
                        fixed_mask,
                        fixed_values,
                        lower,
                        upper,
                        free_idx,
                    )

    _sort_simplex(simplex, scores)
    for j in range(3):
        full_best[j] = fixed_values[j] if fixed_mask[j] else 0.0
    for j in range(free_count):
        full_best[free_idx[j]] = simplex[0, j]
    best_score = _dotm_sse(full_best, y, fixed_mask, fixed_values, lower, upper)
    return full_best, best_score


@njit(cache=True, fastmath=True)
def _free_score(free_params, y, fixed_mask, fixed_values, lower, upper, free_idx):
    initial_level = fixed_values[0] if fixed_mask[0] else 0.0
    alpha = fixed_values[1] if fixed_mask[1] else 0.0
    theta = fixed_values[2] if fixed_mask[2] else 0.0
    for i in range(free_idx.shape[0]):
        if free_idx[i] == 0:
            initial_level = free_params[i]
        elif free_idx[i] == 1:
            alpha = free_params[i]
        else:
            theta = free_params[i]
    if (
        not np.isfinite(initial_level)
        or initial_level < lower[0]
        or initial_level > upper[0]
    ):
        return 1e300
    if not np.isfinite(alpha) or alpha < lower[1] or alpha > upper[1]:
        return 1e300
    if not np.isfinite(theta) or theta < lower[2] or theta > upper[2]:
        return 1e300
    if alpha <= 0.0 or alpha >= 1.0 or theta < 1.0:
        return 1e300
    return _dotm_sse_values(initial_level, alpha, theta, y)


@njit(cache=True, fastmath=True)
def _sort_simplex(simplex, scores):
    n = scores.shape[0]
    for i in range(1, n):
        score = scores[i]
        vertex = simplex[i].copy()
        j = i - 1
        while j >= 0 and scores[j] > score:
            scores[j + 1] = scores[j]
            simplex[j + 1] = simplex[j]
            j -= 1
        scores[j + 1] = score
        simplex[j + 1] = vertex


@njit(cache=True, fastmath=True)
def _simplex_converged(simplex, scores, tol):
    best = scores[0]
    max_score_diff = 0.0
    for i in range(1, scores.shape[0]):
        diff = abs(scores[i] - best)
        if diff > max_score_diff:
            max_score_diff = diff
    max_vertex_diff = 0.0
    for i in range(1, simplex.shape[0]):
        for j in range(simplex.shape[1]):
            diff = abs(simplex[i, j] - simplex[0, j])
            if diff > max_vertex_diff:
                max_vertex_diff = diff
    return max_score_diff <= tol and max_vertex_diff <= tol
