"""Complex Exponential Smoothing (CES) forecaster (non-seasonal, Phase 1)."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["CES"]

import numpy as np
from numba import njit
from scipy.optimize import minimize

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class CES(BaseForecaster, IterativeForecastingMixin):
    r"""Complex Exponential Smoothing (CES) forecaster, non-seasonal.

    Implements the non-seasonal Complex Exponential Smoothing model proposed
    by Svetunkov and Kourentzes [1]_. CES extends exponential smoothing by
    using a complex-valued smoothing parameter
    :math:`\tilde{\alpha} = \alpha_0 + i\,\alpha_1` and a two-component state
    :math:`(\ell_{1,t}, \ell_{2,t})` representing the real and imaginary
    parts. Only :math:`\ell_{1,t}` enters the observation equation, while
    :math:`\ell_{2,t}` acts as an internal correction channel.

    State-space form (no exogenous variables, no seasonality):

    * Observation: :math:`\hat{y}_t = \ell_{1,t-1}` with error
      :math:`\varepsilon_t = y_t - \hat{y}_t`.
    * State update:

      .. math::

          \ell_{1,t} &= \ell_{1,t-1} + (\alpha_1 - 1)\,\ell_{2,t-1}
                       + (\alpha_0 - \alpha_1)\,\varepsilon_t \\
          \ell_{2,t} &= \ell_{1,t-1} + (1 - \alpha_0)\,\ell_{2,t-1}
                       + (\alpha_0 + \alpha_1)\,\varepsilon_t

    * Multi-step forecasts iterate the state forward with zero error and
      return :math:`\ell_{1,n+h-1}` as the :math:`h`-step-ahead forecast.

    The implementation matches the matrix form used by the R ``smooth::ces``
    package for the ``seasonality = "none"`` case.

    Phase 1 caveats:

    * Only the non-seasonal model is implemented.
    * Exogenous variables and prediction intervals are not supported.
    * Both initial state components are tied to ``initial_level`` to keep the
      public API to a single initial parameter; a separate imaginary init
      may be added later.

    Parameters
    ----------
    alpha_real : float or None, default=None
        Real part of the complex smoothing parameter (:math:`\alpha_0`). If
        ``None`` it is estimated.
    alpha_imag : float or None, default=None
        Imaginary part of the complex smoothing parameter (:math:`\alpha_1`).
        If ``None`` it is estimated.
    initial_level : float or None, default=None
        Initial value for both real and imaginary state components. If
        ``None`` it is estimated.
    alpha_real_bounds : tuple of float, default=(0.0, 1.0)
        Bounds for the real part of the smoothing parameter.
    alpha_imag_bounds : tuple of float, default=(-1.0, 1.0)
        Bounds for the imaginary part of the smoothing parameter.
    initial_level_bounds : tuple of float, default=(-1e10, 1e10)
        Bounds for the initial level.
    max_iter : int, default=500
        Maximum number of optimiser iterations.
    tol : float, default=1e-6
        Optimiser convergence tolerance.

    Attributes
    ----------
    alpha_real_ : float
        Fitted real part of the complex smoothing parameter.
    alpha_imag_ : float
        Fitted imaginary part of the complex smoothing parameter.
    complex_alpha_ : complex
        ``alpha_real_ + 1j * alpha_imag_`` for convenience.
    initial_level_ : float
        Fitted initial state value (used for both components).
    level_real_ : float
        Final value of the real state component after fitting.
    level_imag_ : float
        Final value of the imaginary state component after fitting.
    fitted_values_ : np.ndarray
        In-sample one-step-ahead forecasts on the original scale.
    residuals_ : np.ndarray
        In-sample residuals (``y - fitted_values_``).
    forecast_ : float
        Stored one-step-ahead forecast computed at fit time.
    sse_ : float
        Sum of squared in-sample residuals.

    References
    ----------
    .. [1] Svetunkov, I. and Kourentzes, N. (2018). Complex exponential
       smoothing for time series forecasting. Naval Research Logistics,
       65(8), 685-704.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting.stats import CES
    >>> y = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.4, 4.9, 5.3, 5.9])
    >>> forecaster = CES()
    >>> pred = forecaster.iterative_forecast(y, prediction_horizon=2)
    >>> pred.shape
    (2,)
    """

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
        "python_dependencies": None,
    }

    def __init__(
        self,
        alpha_real=None,
        alpha_imag=None,
        initial_level=None,
        alpha_real_bounds=(0.0, 1.0),
        alpha_imag_bounds=(-1.0, 1.0),
        initial_level_bounds=(-1e10, 1e10),
        max_iter=500,
        tol=1e-6,
    ):
        self.alpha_real = alpha_real
        self.alpha_imag = alpha_imag
        self.initial_level = initial_level
        self.alpha_real_bounds = alpha_real_bounds
        self.alpha_imag_bounds = alpha_imag_bounds
        self.initial_level_bounds = initial_level_bounds
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit non-seasonal CES to a univariate series."""
        if exog is not None:
            raise NotImplementedError("CES does not support exogenous variables.")

        y = _prepare_ces_y(y)
        self._validate_ces_params()

        # Build initial parameter vector and bounds for the free parameters.
        x0, lower, upper, fixed_mask, fixed_values = self._build_optimisation_inputs(y)

        scale = float(np.mean(np.abs(y)))
        if scale <= 1e-12 or not np.isfinite(scale):
            scale = 1.0

        if x0.shape[0] == 0:
            # All parameters fixed by the user — no optimisation, just evaluate.
            params = fixed_values
        else:

            def objective(free):
                full = _expand_params(free, fixed_mask, fixed_values)
                return _ces_scaled_sse(full[0], full[1], full[2], y, scale)

            free_bounds = list(zip(lower.tolist(), upper.tolist()))
            # Nelder-Mead with bounds — matches the optimiser family used by
            # the R smooth::ces reference implementation. Derivative-free,
            # robust on the non-smooth admissibility-region boundary.
            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=free_bounds,
                options={
                    "maxiter": int(self.max_iter),
                    "xatol": float(self.tol),
                    "fatol": float(self.tol),
                },
            )
            params = _expand_params(result.x, fixed_mask, fixed_values)

        alpha_0 = float(params[0])
        alpha_1 = float(params[1])
        init_level = float(params[2])

        fitted, residuals, l1_final, l2_final = _ces_recursion(
            y, alpha_0, alpha_1, init_level, init_level
        )

        self.alpha_real_ = alpha_0
        self.alpha_imag_ = alpha_1
        self.complex_alpha_ = alpha_0 + 1j * alpha_1
        self.initial_level_ = init_level
        self.level_real_ = float(l1_final)
        self.level_imag_ = float(l2_final)
        self.fitted_values_ = fitted
        self.residuals_ = residuals
        self.sse_ = float(np.sum(residuals * residuals))

        forecast = _ces_forecast_from_state(1, l1_final, l2_final, alpha_0, alpha_1)[0]
        self.forecast_ = float(forecast)

        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead from the supplied context ``y``."""
        if exog is not None:
            raise NotImplementedError("CES does not support exogenous variables.")
        y = _prepare_ces_y(y, min_length=1)
        # Replay the recurrence on the supplied context using fitted parameters,
        # then take the next one-step forecast from the resulting state.
        _, _, l1, l2 = _ces_recursion(
            y,
            self.alpha_real_,
            self.alpha_imag_,
            self.initial_level_,
            self.initial_level_,
        )
        return float(
            _ces_forecast_from_state(1, l1, l2, self.alpha_real_, self.alpha_imag_)[0]
        )

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit CES on ``y`` and produce ``prediction_horizon`` step forecasts.

        ``exog`` and ``future_exog`` are accepted for signature compatibility
        with :class:`~aeon.forecasting.base.IterativeForecastingMixin` but are
        not yet supported by CES; passing either raises
        :class:`NotImplementedError`. Exogenous-variable support is planned
        for a later phase.
        """
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        if exog is not None or future_exog is not None:
            raise NotImplementedError("CES does not support exogenous variables.")
        self.fit(y)
        return _ces_forecast_from_state(
            int(prediction_horizon),
            self.level_real_,
            self.level_imag_,
            self.alpha_real_,
            self.alpha_imag_,
        )

    def _validate_ces_params(self):
        """Validate parameter bounds and fixed parameter values."""
        lo_a0, hi_a0 = self.alpha_real_bounds
        lo_a1, hi_a1 = self.alpha_imag_bounds
        lo_l0, hi_l0 = self.initial_level_bounds
        if not (np.isfinite(lo_a0) and np.isfinite(hi_a0) and lo_a0 <= hi_a0):
            raise ValueError("alpha_real_bounds must be finite with lower <= upper.")
        if not (np.isfinite(lo_a1) and np.isfinite(hi_a1) and lo_a1 <= hi_a1):
            raise ValueError("alpha_imag_bounds must be finite with lower <= upper.")
        if not (np.isfinite(lo_l0) and np.isfinite(hi_l0) and lo_l0 <= hi_l0):
            raise ValueError("initial_level_bounds must be finite with lower <= upper.")
        for name, val, lo, hi in (
            ("alpha_real", self.alpha_real, lo_a0, hi_a0),
            ("alpha_imag", self.alpha_imag, lo_a1, hi_a1),
            ("initial_level", self.initial_level, lo_l0, hi_l0),
        ):
            if val is None:
                continue
            v = float(val)
            if not np.isfinite(v) or v < lo or v > hi:
                raise ValueError(
                    f"Fixed {name}={val!r} is not finite or lies outside its bounds."
                )

    def _build_optimisation_inputs(self, y):
        """Build the free-parameter initial point and bounds for the optimiser."""
        defaults = (0.5, 0.0, float(np.mean(y)))
        user_values = (self.alpha_real, self.alpha_imag, self.initial_level)
        full_bounds = (
            self.alpha_real_bounds,
            self.alpha_imag_bounds,
            self.initial_level_bounds,
        )
        fixed_mask = np.zeros(3, dtype=np.bool_)
        fixed_values = np.empty(3, dtype=np.float64)
        free_x0 = []
        free_lower = []
        free_upper = []
        for i, (val, default, (lo, hi)) in enumerate(
            zip(user_values, defaults, full_bounds)
        ):
            if val is not None:
                fixed_mask[i] = True
                fixed_values[i] = float(val)
            else:
                fixed_mask[i] = False
                fixed_values[i] = np.nan
                x0 = min(max(default, lo), hi)
                free_x0.append(x0)
                free_lower.append(lo)
                free_upper.append(hi)
        return (
            np.asarray(free_x0, dtype=np.float64),
            np.asarray(free_lower, dtype=np.float64),
            np.asarray(free_upper, dtype=np.float64),
            fixed_mask,
            fixed_values,
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {"alpha_real": 0.5, "alpha_imag": 0.0, "initial_level": 0.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_ces_y(y, min_length=2):
    """Validate and coerce ``y`` to a 1D float64 array."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] < min_length:
        raise ValueError(f"CES requires at least {min_length} observations.")
    if not np.all(np.isfinite(y)):
        raise ValueError("CES requires finite values.")
    return y


def _expand_params(free, fixed_mask, fixed_values):
    """Combine free-parameter vector with fixed values into a length-3 array."""
    out = np.empty(3, dtype=np.float64)
    j = 0
    for i in range(3):
        if fixed_mask[i]:
            out[i] = fixed_values[i]
        else:
            out[i] = free[j]
            j += 1
    return out


@njit(cache=True, fastmath=True)
def _ces_recursion(y, alpha_0, alpha_1, l1_0, l2_0):
    """Run the CES state recurrence over ``y``.

    Returns
    -------
    fitted : np.ndarray
        One-step-ahead in-sample fitted values.
    residuals : np.ndarray
        ``y - fitted``.
    l1_final, l2_final : float
        Final real and imaginary state components after consuming ``y``.
    """
    n = y.shape[0]
    fitted = np.empty(n, dtype=np.float64)
    residuals = np.empty(n, dtype=np.float64)
    l1 = l1_0
    l2 = l2_0
    f12 = alpha_1 - 1.0
    f22 = 1.0 - alpha_0
    g1 = alpha_0 - alpha_1
    g2 = alpha_0 + alpha_1

    for t in range(n):
        yhat = l1
        fitted[t] = yhat
        eps = y[t] - yhat
        residuals[t] = eps
        new_l1 = l1 + f12 * l2 + g1 * eps
        new_l2 = l1 + f22 * l2 + g2 * eps
        l1 = new_l1
        l2 = new_l2

    return fitted, residuals, l1, l2


@njit(cache=True, fastmath=True)
def _ces_forecast_from_state(h, l1, l2, alpha_0, alpha_1):
    """Project ``h`` CES forecasts forward from a final state (zero error)."""
    forecast = np.empty(h, dtype=np.float64)
    f12 = alpha_1 - 1.0
    f22 = 1.0 - alpha_0
    for k in range(h):
        forecast[k] = l1
        new_l1 = l1 + f12 * l2
        new_l2 = l1 + f22 * l2
        l1 = new_l1
        l2 = new_l2
    return forecast


@njit(cache=True, fastmath=True)
def _ces_scaled_sse(alpha_0, alpha_1, init_level, y, scale):
    """Scaled in-sample SSE objective for CES parameter optimisation."""
    n = y.shape[0]
    l1 = init_level
    l2 = init_level
    f12 = alpha_1 - 1.0
    f22 = 1.0 - alpha_0
    g1 = alpha_0 - alpha_1
    g2 = alpha_0 + alpha_1
    sse = 0.0
    for t in range(n):
        yhat = l1
        eps = y[t] - yhat
        if not np.isfinite(yhat) or not np.isfinite(eps):
            return 1e300
        err = eps / scale
        sse += err * err
        new_l1 = l1 + f12 * l2 + g1 * eps
        new_l2 = l1 + f22 * l2 + g2 * eps
        l1 = new_l1
        l2 = new_l2
    if not np.isfinite(sse):
        return 1e300
    return sse
