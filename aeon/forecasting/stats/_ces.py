"""Complex Exponential Smoothing (CES) forecasters."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["CES", "AutoCES"]

import math

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin

# Public model codes, plus accepted long-form aliases.
_MODEL_ALIASES = {
    "N": "N",
    "none": "N",
    "S": "S",
    "simple": "S",
    "P": "P",
    "partial": "P",
    "F": "F",
    "full": "F",
}
_IMPLEMENTED_MODELS = ("N", "S", "P", "F")
_VALID_IC = ("aic", "aicc", "bic")
_CES_NONE = 0
_CES_SIMPLE = 1
_CES_PARTIAL = 2
_CES_FULL = 3
_MODEL_TO_SEASON = {
    "N": _CES_NONE,
    "S": _CES_SIMPLE,
    "P": _CES_PARTIAL,
    "F": _CES_FULL,
}
_MODEL_COMPONENTS = {
    "N": 2,
    "S": 2,
    "P": 3,
    "F": 4,
}


def _validate_season_length(value):
    """Reject silent coercion of bad ``season_length`` values.

    ``season_length`` must be a strictly positive integer. Booleans
    (which Python classifies as ``int``) are rejected explicitly because
    ``True`` quietly coerces to ``1`` and is almost never user-intended.
    Floats and strings are also rejected even when they look numeric:
    ``int(4.9) == 4`` and ``int("4") == 4`` are silent rounding/parsing
    that mask user typos.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"season_length must be a positive integer, got {value!r}.")


def _normalise_model(model):
    """Resolve a user-facing model code or alias to its canonical letter."""
    if not isinstance(model, str):
        raise ValueError(
            f"model must be a string, got {type(model).__name__}: {model!r}."
        )
    key = model.lower() if model.lower() in _MODEL_ALIASES else model
    if key not in _MODEL_ALIASES:
        raise ValueError(
            f"Unknown CES model {model!r}. Expected one of "
            f"{sorted(set(_MODEL_ALIASES))}."
        )
    return _MODEL_ALIASES[key]


class CES(BaseForecaster, IterativeForecastingMixin):
    r"""Complex Exponential Smoothing (CES) forecaster.

    Implements the CES family proposed by Svetunkov and Kourentzes [1]_.
    CES extends exponential smoothing by using complex-valued smoothing
    parameters and two-component (real + imaginary correction) state.

    Four model variants are implemented:

    * ``model="N"`` (non-seasonal): a single two-component state
      :math:`(\ell_{1,t}, \ell_{2,t})` with complex smoothing parameter
      :math:`\tilde{\alpha} = \alpha_0 + i\,\alpha_1`. The recurrence is
      :math:`\hat{y}_t = \ell_{1,t-1}`,
      :math:`\ell_{1,t} = \ell_{1,t-1} + (\alpha_1-1)\,\ell_{2,t-1}
      + (\alpha_0-\alpha_1)\,\varepsilon_t`,
      :math:`\ell_{2,t} = \ell_{1,t-1} + (1-\alpha_0)\,\ell_{2,t-1}
      + (\alpha_0+\alpha_1)\,\varepsilon_t`.
      The persistence vector ``(\alpha_0 - \alpha_1, \alpha_0 + \alpha_1)``
      matches the smooth/ADAM and StatsForecast AutoCES parameterisation
      so fitted alphas transfer directly between libraries.
    * ``model="S"`` (simple seasonal): a two-component state is updated at
      seasonal lag ``season_length``. The one-step forecast uses the state
      from the same position in the previous seasonal cycle.
    * ``model="P"`` (partial seasonal): combines the non-seasonal
      two-component state with a one-component additive seasonal state
      updated at seasonal lag ``season_length``.
    * ``model="F"`` (full seasonal): adds a seasonal complex-state pair
      :math:`(l_{1,t}, c_{1,t})` lagged by ``season_length`` and a second
      complex smoothing parameter :math:`\tilde{\beta} = \beta_0 + i\,\beta_1`.
      The observation is
      :math:`\hat{y}_t = l_{0,t-1} + l_{1,t-m}`, and the seasonal state has
      its own CES-style update driven by the same error series.

    The estimator uses deterministic initial states and a backfitting pass
    following StatsForecast's ``AutoCES`` implementation, so aeon and
    StatsForecast forecasts can be compared directly for ``N/S/P/F`` and
    AutoCES ``model="Z"`` style selection.

    Parameters
    ----------
    model : {"N", "S", "P", "F", "none", "simple", "partial", "full"}
        Model code. ``"N"`` is the default. Seasonal models require
        ``season_length >= 2``.
    season_length : int, default=1
        Seasonal period. ``1`` means no seasonality. Required > 1 for any
        seasonal model.
    alpha_real, alpha_imag : float or None, default=None
        Components of the non-seasonal complex smoothing parameter. If
        ``None`` they are estimated.
    beta_real, beta_imag : float or None, default=None
        Components of the seasonal smoothing parameter. ``beta_real`` is
        used by ``"P"`` and ``"F"``; ``beta_imag`` is used only by ``"F"``.
        If ``None`` the relevant components are estimated.
    initial_level : float or None, default=None
        Initial value for the non-seasonal real-state component
        ``ell_{1,0}`` / ``l_{0,0}``. If ``None`` it is seeded
        deterministically from the data.
    initial_level_imag : float or None, default=None
        Initial value for the non-seasonal imaginary-state component
        ``ell_{2,0}`` / ``c_{0,0}``. If ``None`` it is seeded
        deterministically from the data.
    initial_seasonal_real, initial_seasonal_imag : array-like or None
        Length-``season_length`` initial seasonal state. If ``None`` the
        seasonal seed is computed deterministically from the data (per-
        position mean minus overall mean; correction set to zero). The
        seasonal seed is deterministic and is not optimised.
    alpha_real_bounds : tuple of float, default=(0.01, 1.8)
        Box bounds for the optimisation of ``alpha_real``. Match
        StatsForecast's AutoCES non-seasonal optimiser bounds. These are
        practical optimisation bounds, not a complete CES admissibility
        region.
    alpha_imag_bounds : tuple of float, default=(0.01, 1.9)
        Box bounds for the optimisation of ``alpha_imag``.
    beta_real_bounds, beta_imag_bounds : tuple of float
        Box bounds for the seasonal smoothing parameters; default to
        StatsForecast's seasonal CES bounds.
    initial_level_bounds : tuple of float, default=(-1e10, 1e10)
        Box bounds for both real and imaginary initial-state components.
    max_iter : int, default=1000
        Maximum number of optimiser iterations.
    tol : float, default=1e-4
        Optimiser convergence tolerance.

    Attributes
    ----------
    model_ : str
        Canonical model code actually used (``"N"``, ``"S"``, ``"P"``,
        or ``"F"``).
    season_length_ : int
        Seasonal period used (1 for non-seasonal).
    alpha_real_, alpha_imag_ : float
        Fitted non-seasonal smoothing parameters.
    complex_alpha_ : complex
        Convenience ``alpha_real_ + 1j * alpha_imag_``.
    beta_real_, beta_imag_ : float
        Fitted seasonal smoothing parameters (``NaN`` for non-seasonal).
    complex_beta_ : complex
        Convenience ``beta_real_ + 1j * beta_imag_`` (``NaN`` for non-seasonal).
    initial_level_, initial_level_imag_ : float
        Fitted initial non-seasonal state components.
    initial_seasonal_real_, initial_seasonal_imag_ : np.ndarray or None
        Seasonal state seed (length ``season_length_``). ``None`` for
        non-seasonal.
    level_real_, level_imag_ : float
        Final non-seasonal state components after fitting.
    seasonal_real_, seasonal_imag_ : np.ndarray or None
        Final seasonal-state buffers after fitting (length ``season_length_``).
        ``None`` for non-seasonal.
    fitted_values_ : np.ndarray
        In-sample one-step-ahead fitted values.
    residuals_ : np.ndarray
        ``y - fitted_values_``.
    forecast_ : float
        Stored one-step-ahead forecast.
    sse_ : float
        Sum of squared in-sample residuals.
    n_params_ : int
        Number of parameters used for IC calculations.
    n_free_params_ : int
        Number of smoothing parameters optimised by the fit.
    optimization_success_ : bool
        Optimiser success flag.
    optimization_message_ : str
        Optimiser termination message.
    n_iter_ : int
        Optimiser iteration count.

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
    >>> f = CES()
    >>> f.iterative_forecast(y, prediction_horizon=2).round(4)
    array([6.383 , 7.0506])
    """

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
        "python_dependencies": None,
    }

    def __init__(
        self,
        model="N",
        season_length=1,
        alpha_real=None,
        alpha_imag=None,
        beta_real=None,
        beta_imag=None,
        initial_level=None,
        initial_level_imag=None,
        initial_seasonal_real=None,
        initial_seasonal_imag=None,
        # Bounds match StatsForecast's AutoCES optimiser bounds for both
        # alpha and beta components. These are practical optimisation
        # bounds, not the complete CES admissibility region.
        alpha_real_bounds=(0.01, 1.8),
        alpha_imag_bounds=(0.01, 1.9),
        beta_real_bounds=(0.01, 1.5),
        beta_imag_bounds=(0.01, 1.5),
        initial_level_bounds=(-1e10, 1e10),
        max_iter=1000,
        tol=1e-4,
    ):
        self.model = model
        self.season_length = season_length
        self.alpha_real = alpha_real
        self.alpha_imag = alpha_imag
        self.beta_real = beta_real
        self.beta_imag = beta_imag
        self.initial_level = initial_level
        self.initial_level_imag = initial_level_imag
        self.initial_seasonal_real = initial_seasonal_real
        self.initial_seasonal_imag = initial_seasonal_imag
        self.alpha_real_bounds = alpha_real_bounds
        self.alpha_imag_bounds = alpha_imag_bounds
        self.beta_real_bounds = beta_real_bounds
        self.beta_imag_bounds = beta_imag_bounds
        self.initial_level_bounds = initial_level_bounds
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(horizon=1, axis=1)

    # ------------------------------------------------------------------
    # fit / predict / iterative_forecast
    # ------------------------------------------------------------------

    def _fit(self, y, exog=None):
        """Fit CES to a univariate series.

        Dispatches on the requested model code and fits the StatsForecast-style
        CES state-array recurrence. All four CES variants (``"N"``, ``"S"``,
        ``"P"``, ``"F"``) use deterministic initial states and optimise only
        the smoothing parameters, matching the reference implementation's
        model-comparison setup.
        """
        if exog is not None:
            raise NotImplementedError("CES does not support exogenous variables.")

        model_code = _normalise_model(self.model)
        self._check_seasonal_settings(model_code)
        y = _prepare_ces_y(y)
        self._validate_ces_param_bounds(model_code)

        self.model_ = model_code
        self.season_length_ = int(self.season_length) if model_code != "N" else 1
        self._fit_statsforecast_style(y)
        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead from the supplied context ``y``."""
        if exog is not None:
            raise NotImplementedError("CES does not support exogenous variables.")
        y = _prepare_ces_y(y, min_length=1)
        m = self._fit_m_
        season = self._season_code_
        _, _, states, _ = _ces_fit_states(
            y,
            m,
            season,
            self.alpha_real_,
            self.alpha_imag_,
            self.beta_real_,
            self.beta_imag_,
            self._initial_states_,
            True,
        )
        return float(
            _ces_one_step_forecast_from_states(
                states,
                int(y.shape[0]),
                m,
                season,
            )
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
        with :class:`~aeon.forecasting.base.IterativeForecastingMixin` only;
        passing either raises :class:`NotImplementedError`.
        """
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        if exog is not None or future_exog is not None:
            raise NotImplementedError("CES does not support exogenous variables.")
        self.fit(y)
        h = int(prediction_horizon)
        return _ces_forecast_from_states(
            h,
            self._states_,
            int(self._n_train_),
            self._fit_m_,
            self._season_code_,
            self.alpha_real_,
            self.alpha_imag_,
            self.beta_real_,
            self.beta_imag_,
        )

    # ------------------------------------------------------------------
    # Per-model fit helpers
    # ------------------------------------------------------------------

    def _fit_statsforecast_style(self, y):
        """Fit a CES variant using StatsForecast-compatible state semantics."""
        model_code = self.model_
        m = int(self.season_length_) if model_code != "N" else 1
        if model_code != "N" and y.shape[0] < 2 * m:
            raise ValueError(
                f"Seasonal CES (model={model_code!r}) needs at least "
                f"2*season_length observations; got n={y.shape[0]}, "
                f"season_length={m}."
            )

        season = _MODEL_TO_SEASON[model_code]
        init_states = self._build_initial_states(y, m, model_code)
        x0, lower, upper, fixed_mask, fixed_values = (
            self._build_optimisation_inputs_model(model_code)
        )

        if x0.shape[0] == 0:
            params = fixed_values.copy()
            success, message, n_iter = True, "All parameters fixed.", 0
        else:
            free_params, n_iter, converged = _run_ces_nelder_mead(
                x0,
                lower,
                upper,
                fixed_mask,
                fixed_values,
                y,
                m,
                season,
                init_states,
                self.max_iter,
                self.tol,
            )
            params = _expand_params(free_params, fixed_mask, fixed_values)
            success = bool(converged)
            message = (
                "Optimization converged." if success else "Maximum iterations reached."
            )

        alpha_0, alpha_1, beta_0, beta_1 = _unpack_ces_params(params, model_code)
        fitted, residuals, states, likelihood_objective = _ces_fit_states(
            y, m, season, alpha_0, alpha_1, beta_0, beta_1, init_states, True
        )
        forecast = _ces_one_step_forecast_from_states(
            states,
            int(y.shape[0]),
            m,
            season,
        )

        self.alpha_real_ = float(alpha_0)
        self.alpha_imag_ = float(alpha_1)
        self.complex_alpha_ = alpha_0 + 1j * alpha_1
        self.beta_real_ = float(beta_0)
        self.beta_imag_ = float(beta_1)
        self.complex_beta_ = beta_0 + 1j * beta_1
        self.initial_level_ = float(init_states[0, 0])
        self.initial_level_imag_ = float(init_states[0, 1])
        self.initial_seasonal_real_, self.initial_seasonal_imag_ = (
            _extract_seasonal_attrs(init_states, model_code)
        )
        final_buffer = states[int(y.shape[0]) : int(y.shape[0]) + m]
        self.level_real_ = float(final_buffer[-1, 0])
        self.level_imag_ = float(final_buffer[-1, 1])
        self.seasonal_real_, self.seasonal_imag_ = _extract_seasonal_attrs(
            final_buffer, model_code
        )
        self.fitted_values_ = fitted
        self.residuals_ = residuals
        self.sse_ = float(np.sum(residuals * residuals))
        self.loglikelihood_objective_ = float(likelihood_objective)
        self.forecast_ = float(forecast)
        self.n_params_ = _MODEL_COMPONENTS[model_code] + 1
        self.n_free_params_ = int(np.sum(~fixed_mask))
        self.optimization_success_ = success
        self.optimization_message_ = message
        self.n_iter_ = n_iter
        self._n_train_ = int(y.shape[0])
        self._fit_m_ = m
        self._season_code_ = season
        self._initial_states_ = init_states.copy()
        self._states_ = states.copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_seasonal_settings(self, model_code):
        """Cross-check ``season_length`` against ``model``."""
        _validate_season_length(self.season_length)
        if model_code != "N" and self.season_length < 2:
            raise ValueError(
                f"Seasonal CES (model={model_code!r}) requires " "season_length >= 2."
            )

    def _validate_ces_param_bounds(self, model_code):
        """Validate bounds and fixed-value feasibility for the chosen model."""
        param_specs = [
            ("alpha_real", self.alpha_real, self.alpha_real_bounds),
            ("alpha_imag", self.alpha_imag, self.alpha_imag_bounds),
            ("initial_level", self.initial_level, self.initial_level_bounds),
            ("initial_level_imag", self.initial_level_imag, self.initial_level_bounds),
        ]
        if model_code in ("P", "F"):
            param_specs.append(("beta_real", self.beta_real, self.beta_real_bounds))
        if model_code == "F":
            param_specs.extend(
                [
                    ("beta_imag", self.beta_imag, self.beta_imag_bounds),
                ]
            )
        for name, val, bounds in param_specs:
            lo, hi = bounds
            if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
                raise ValueError(
                    f"{name}_bounds must be finite with lower <= upper, got {bounds!r}."
                )
            if val is None:
                continue
            v = float(val)
            if not np.isfinite(v) or v < lo or v > hi:
                raise ValueError(
                    f"Fixed {name}={val!r} is not finite or lies outside its bounds."
                )

    # ------------------------------------------------------------------
    # Optimisation-input builders
    # ------------------------------------------------------------------

    def _build_optimisation_inputs_model(self, model_code):
        """Build initial point, bounds and fixed mask for StatsForecast-style fit."""
        if model_code in ("N", "S"):
            return _pack_optimisation_inputs(
                defaults=(1.3, 1.0),
                user_values=(self.alpha_real, self.alpha_imag),
                full_bounds=(self.alpha_real_bounds, self.alpha_imag_bounds),
            )
        if model_code == "P":
            return _pack_optimisation_inputs(
                defaults=(1.3, 1.0, 0.1),
                user_values=(self.alpha_real, self.alpha_imag, self.beta_real),
                full_bounds=(
                    self.alpha_real_bounds,
                    self.alpha_imag_bounds,
                    self.beta_real_bounds,
                ),
            )
        return _pack_optimisation_inputs(
            defaults=(1.3, 1.0, 1.3, 1.0),
            user_values=(
                self.alpha_real,
                self.alpha_imag,
                self.beta_real,
                self.beta_imag,
            ),
            full_bounds=(
                self.alpha_real_bounds,
                self.alpha_imag_bounds,
                self.beta_real_bounds,
                self.beta_imag_bounds,
            ),
        )

    def _build_initial_states(self, y, m, model_code):
        """Build deterministic initial states matching StatsForecast AutoCES."""
        components = _MODEL_COMPONENTS[model_code]
        lags = 1 if model_code == "N" else m
        states = np.zeros((lags, components), dtype=np.float64)

        if model_code == "N":
            idx = min(max(10, m), y.shape[0])
            mean = float(np.mean(y[:idx]))
            states[0, 0] = mean
            states[0, 1] = mean / 1.1
        elif model_code == "S":
            states[:lags, 0] = y[:lags]
            states[:lags, 1] = y[:lags] / 1.1
        else:
            mean = float(np.mean(y[:lags]))
            states[:lags, 0] = mean
            states[:lags, 1] = mean / 1.1
            seasonal = _seasonal_decompose_additive(y, lags)[:lags]
            states[:lags, 2] = seasonal
            if model_code == "F":
                states[:lags, 3] = seasonal / 1.1

        if self.initial_level is not None:
            states[:, 0] = float(self.initial_level)
        if self.initial_level_imag is not None:
            states[:, 1] = float(self.initial_level_imag)

        if model_code in ("P", "F") and self.initial_seasonal_real is not None:
            real_seed = np.asarray(self.initial_seasonal_real, dtype=np.float64)
            if real_seed.shape != (lags,):
                raise ValueError(
                    "initial_seasonal_real must have length "
                    f"season_length ({lags}), got shape {real_seed.shape}."
                )
            states[:, 2] = real_seed
        if model_code == "F" and self.initial_seasonal_imag is not None:
            imag_seed = np.asarray(self.initial_seasonal_imag, dtype=np.float64)
            if imag_seed.shape != (lags,):
                raise ValueError(
                    "initial_seasonal_imag must have length "
                    f"season_length ({lags}), got shape {imag_seed.shape}."
                )
            states[:, 3] = imag_seed

        if not np.all(np.isfinite(states)):
            raise ValueError("Initial CES states contain non-finite values.")
        return states

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {"alpha_real": 0.5, "alpha_imag": 0.5, "initial_level": 0.0}


# ---------------------------------------------------------------------------
# AutoCES
# ---------------------------------------------------------------------------


class AutoCES(BaseForecaster, IterativeForecastingMixin):
    """Automatic CES selector by information criterion.

    Fits each candidate :class:`CES` model in turn, then picks the one with
    the lowest information criterion (AICc by default). Seasonal candidates
    are silently skipped when ``season_length <= 1`` or the series is too
    short for the seasonal seed.

    Parameters
    ----------
    season_length : int, default=1
        Seasonal period passed to seasonal candidates.
    models : sequence of str, default=("N", "S", "P", "F")
        Candidate CES model codes to consider. Defaults to the StatsForecast
        ``model="Z"`` search set.
    ic : {"aic", "aicc", "bic"}, default="aicc"
        Information criterion used for model selection.
    max_iter : int, default=1000
        Maximum optimiser iterations per candidate.
    tol : float, default=1e-4
        Optimiser convergence tolerance per candidate.

    Attributes
    ----------
    best_model_ : CES
        The fitted :class:`CES` instance selected by IC.
    best_model_name_ : str
        Canonical code of the selected model (e.g. ``"N"`` or ``"F"``).
    model_results_ : dict
        Per-candidate dictionary keyed by model code with ``sse``, ``k``,
        ``aic``, ``aicc``, ``bic``, ``status`` (``"ok"`` / ``"skipped"`` /
        ``"error"``), and ``message``. Failed candidates do not block
        selection.
    fitted_values_, residuals_, forecast_, sse_ :
        Pass-throughs from ``best_model_`` for convenience.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting.stats import AutoCES
    >>> y = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.4, 4.9, 5.3, 5.9])
    >>> f = AutoCES()
    >>> f.iterative_forecast(y, prediction_horizon=2).round(4)
    array([6.383 , 7.0506])
    """

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": False,
        "python_dependencies": None,
    }

    def __init__(
        self,
        season_length=1,
        models=("N", "S", "P", "F"),
        ic="aicc",
        max_iter=1000,
        tol=1e-4,
    ):
        self.season_length = season_length
        self.models = models
        self.ic = ic
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit each candidate model and pick the one with lowest IC."""
        if exog is not None:
            raise NotImplementedError("AutoCES does not support exogenous variables.")
        if self.ic not in _VALID_IC:
            raise ValueError(f"ic must be one of {_VALID_IC}, got {self.ic!r}.")
        # Resolve each requested model code (lets users pass "full" etc.).
        try:
            requested = tuple(_normalise_model(m) for m in self.models)
        except TypeError as exc:
            raise ValueError("models must be a sequence of strings.") from exc
        if not requested:
            raise ValueError("models must contain at least one candidate.")

        # Reject silent coercion of bad season_length values (floats,
        # strings, booleans, non-positive ints). CES._check_seasonal_settings
        # enforces the same rule for the single-model path; this guard
        # ensures AutoCES does not bypass it on the way to building
        # candidates.
        _validate_season_length(self.season_length)

        y_arr = _prepare_ces_y(y)
        n = int(y_arr.shape[0])
        m = int(self.season_length)

        results = {}
        fits = {}
        for code in requested:
            if code != "N" and m < 2:
                results[code] = _ic_failure(
                    "skipped",
                    f"Seasonal model {code!r} requires season_length >= 2; "
                    f"got {m}.",
                )
                continue
            if code != "N" and n < 2 * m:
                results[code] = _ic_failure(
                    "skipped",
                    f"Seasonal model {code!r} needs at least 2*season_length "
                    f"observations; got n={n}, m={m}.",
                )
                continue
            try:
                fitted = CES(
                    model=code,
                    season_length=m if code != "N" else 1,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                fitted._fit(y_arr)
                fitted.is_fitted = True
            except Exception as exc:  # noqa: BLE001
                results[code] = _ic_failure("error", repr(exc))
                continue
            sse = max(float(fitted.sse_), 1e-12)
            k = int(fitted.n_params_)
            aic, aicc, bic = _information_criteria(n, sse, k)
            results[code] = {
                "sse": sse,
                "k": k,
                "aic": aic,
                "aicc": aicc,
                "bic": bic,
                "status": "ok",
                "message": getattr(fitted, "optimization_message_", ""),
            }
            fits[code] = fitted

        ok_codes = [c for c, r in results.items() if r["status"] == "ok"]
        if not ok_codes:
            raise ValueError(f"All AutoCES candidates failed: {results!r}.")
        ic_key = self.ic
        best_code = min(ok_codes, key=lambda c: results[c][ic_key])
        best = fits[best_code]

        self.best_model_ = best
        self.best_model_name_ = best_code
        self.model_results_ = results
        # Promote the winning model's outputs for convenience.
        self.fitted_values_ = best.fitted_values_
        self.residuals_ = best.residuals_
        self.forecast_ = best.forecast_
        self.sse_ = best.sse_
        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead from the selected ``best_model_``."""
        if exog is not None:
            raise NotImplementedError("AutoCES does not support exogenous variables.")
        return self.best_model_.predict(y)

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Fit AutoCES and produce ``prediction_horizon`` step forecasts."""
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")
        if exog is not None or future_exog is not None:
            raise NotImplementedError("AutoCES does not support exogenous variables.")
        self.fit(y)
        best = self.best_model_
        return _ces_forecast_from_states(
            int(prediction_horizon),
            best._states_,
            int(best._n_train_),
            best._fit_m_,
            best._season_code_,
            best.alpha_real_,
            best.alpha_imag_,
            best.beta_real_,
            best.beta_imag_,
        )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return {"season_length": 1, "models": ("N",)}


# ---------------------------------------------------------------------------
# Module helpers (pure Python / numpy)
# ---------------------------------------------------------------------------


def _prepare_ces_y(y, min_length=2):
    """Validate and coerce ``y`` to a 1D float64 array."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] < min_length:
        raise ValueError(f"CES requires at least {min_length} observations.")
    if not np.all(np.isfinite(y)):
        raise ValueError("CES requires finite values.")
    return y


def _pack_optimisation_inputs(defaults, user_values, full_bounds):
    """Combine user-fixed and free parameters into optimiser-ready arrays.

    Returns ``(x0, lower, upper, fixed_mask, fixed_values)``.
    """
    n = len(defaults)
    fixed_mask = np.zeros(n, dtype=np.bool_)
    fixed_values = np.empty(n, dtype=np.float64)
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


def _expand_params(free, fixed_mask, fixed_values):
    """Reassemble the full parameter vector from the free subset.

    This helper works for any length given by ``fixed_mask``.
    """
    n = fixed_mask.shape[0]
    out = np.empty(n, dtype=np.float64)
    j = 0
    for i in range(n):
        if fixed_mask[i]:
            out[i] = fixed_values[i]
        else:
            out[i] = free[j]
            j += 1
    return out


def _information_criteria(n, sse, k):
    """Gaussian-SSE-based AIC, AICc and BIC."""
    sse = max(sse, 1e-12)
    aic = n * math.log(sse / n) + 2.0 * k
    denom = n - k - 1
    if denom <= 0:
        aicc = math.inf
    else:
        aicc = aic + (2.0 * k * (k + 1.0)) / denom
    bic = n * math.log(sse / n) + k * math.log(n)
    return aic, aicc, bic


def _ic_failure(status, message):
    """Build a uniform failure record for ``AutoCES.model_results_``."""
    return {
        "sse": math.nan,
        "k": 0,
        "aic": math.inf,
        "aicc": math.inf,
        "bic": math.inf,
        "status": status,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Numba kernels — non-seasonal ("N")
# ---------------------------------------------------------------------------


def _unpack_ces_params(params, model_code):
    """Return ``(alpha_0, alpha_1, beta_0, beta_1)`` for a model code."""
    alpha_0 = float(params[0])
    alpha_1 = float(params[1])
    if model_code == "P":
        return alpha_0, alpha_1, float(params[2]), math.nan
    if model_code == "F":
        return alpha_0, alpha_1, float(params[2]), float(params[3])
    return alpha_0, alpha_1, math.nan, math.nan


def _extract_seasonal_attrs(states, model_code):
    """Extract public seasonal-state attributes from a state buffer."""
    if model_code == "P":
        return states[:, 2].copy(), None
    if model_code == "F":
        return states[:, 2].copy(), states[:, 3].copy()
    return None, None


def _ces_fit_objective(y, m, season, alpha_0, alpha_1, beta_0, beta_1, init_states):
    """StatsForecast-style CES objective: final backfit pass ``n * log(SSE)``."""
    if season == _CES_NONE:
        likelihood_objective = _ces_n_fit_objective_numba(
            y, alpha_0, alpha_1, init_states[0, 0], init_states[0, 1]
        )
    else:
        likelihood_objective = _ces_fit_objective_numba(
            y, m, season, alpha_0, alpha_1, beta_0, beta_1, init_states
        )
    if not np.isfinite(likelihood_objective):
        return 1e300
    return float(likelihood_objective)


@njit(cache=True, fastmath=True)
def _clip_to_bounds(x, lower, upper):
    """Clip an optimizer vector to box bounds in place."""
    for i in range(x.shape[0]):
        if x[i] < lower[i]:
            x[i] = lower[i]
        elif x[i] > upper[i]:
            x[i] = upper[i]


@njit(cache=True, fastmath={"contract"})
def _ces_objective_from_free(
    free,
    fixed_mask,
    fixed_values,
    y,
    m,
    season,
    init_states,
):
    """Evaluate the CES objective from a free optimizer vector."""
    n_total = fixed_mask.shape[0]
    params = np.empty(n_total, dtype=np.float64)
    j = 0
    for i in range(n_total):
        if fixed_mask[i]:
            params[i] = fixed_values[i]
        else:
            params[i] = free[j]
            j += 1

    alpha_0 = params[0]
    alpha_1 = params[1]
    beta_0 = np.nan
    beta_1 = np.nan
    if n_total >= 3:
        beta_0 = params[2]
    if n_total >= 4:
        beta_1 = params[3]

    if season == _CES_NONE:
        objective = _ces_n_fit_objective_numba(
            y, alpha_0, alpha_1, init_states[0, 0], init_states[0, 1]
        )
    else:
        objective = _ces_fit_objective_numba(
            y, m, season, alpha_0, alpha_1, beta_0, beta_1, init_states
        )
    if not np.isfinite(objective):
        return 1e300
    return objective


@njit(cache=True, fastmath=True)
def _std_1d(values):
    """Compute standard deviation for a 1D array."""
    n = values.shape[0]
    mean = 0.0
    for i in range(n):
        mean += values[i]
    mean /= n
    var = 0.0
    for i in range(n):
        diff = values[i] - mean
        var += diff * diff
    return (var / n) ** 0.5


@njit(cache=True, fastmath=True)
def _run_ces_nelder_mead(
    x0,
    lower,
    upper,
    fixed_mask,
    fixed_values,
    y,
    m,
    season,
    init_states,
    max_iter,
    tol,
):
    """Run bounded adaptive Nelder-Mead for CES smoothing parameters."""
    n = x0.shape[0]
    simplex = np.empty((n + 1, n), dtype=np.float64)
    for row in range(n + 1):
        for col in range(n):
            simplex[row, col] = x0[col]
    _clip_to_bounds(simplex[n], lower, upper)
    for i in range(n):
        value = x0[i]
        if value == 0.0:
            value = 0.0001
        else:
            value *= 1.05
        simplex[i, i] = value
        _clip_to_bounds(simplex[i], lower, upper)

    f_simplex = np.empty(n + 1, dtype=np.float64)
    for row in range(n + 1):
        f_simplex[row] = _ces_objective_from_free(
            simplex[row], fixed_mask, fixed_values, y, m, season, init_states
        )

    alpha = 1.0
    gamma = 1.0 + 2.0 / n
    rho = 0.75 - 1.0 / (2.0 * n)
    sigma = 1.0 - 1.0 / n
    x_o = np.empty(n, dtype=np.float64)
    x_r = np.empty(n, dtype=np.float64)
    x_e = np.empty(n, dtype=np.float64)
    x_c = np.empty(n, dtype=np.float64)

    converged = False
    n_iter = 0
    for iteration in range(int(max_iter)):
        n_iter = iteration + 1
        order = np.argsort(f_simplex)
        best_idx = order[0]
        worst_idx = order[n]
        second_worst_idx = order[n - 1]

        if _std_1d(f_simplex) < tol:
            converged = True
            break

        for col in range(n):
            total = 0.0
            for pos in range(n):
                total += simplex[order[pos], col]
            x_o[col] = total / n

        for col in range(n):
            x_r[col] = x_o[col] + alpha * (x_o[col] - simplex[worst_idx, col])
        _clip_to_bounds(x_r, lower, upper)
        f_r = _ces_objective_from_free(
            x_r, fixed_mask, fixed_values, y, m, season, init_states
        )

        if f_simplex[best_idx] <= f_r < f_simplex[second_worst_idx]:
            for col in range(n):
                simplex[worst_idx, col] = x_r[col]
            f_simplex[worst_idx] = f_r
            continue

        if f_r < f_simplex[best_idx]:
            for col in range(n):
                x_e[col] = x_o[col] + gamma * (x_r[col] - x_o[col])
            _clip_to_bounds(x_e, lower, upper)
            f_e = _ces_objective_from_free(
                x_e, fixed_mask, fixed_values, y, m, season, init_states
            )
            if f_e < f_r:
                for col in range(n):
                    simplex[worst_idx, col] = x_e[col]
                f_simplex[worst_idx] = f_e
            else:
                for col in range(n):
                    simplex[worst_idx, col] = x_r[col]
                f_simplex[worst_idx] = f_r
            continue

        if f_simplex[second_worst_idx] <= f_r < f_simplex[worst_idx]:
            for col in range(n):
                x_c[col] = x_o[col] + rho * (x_r[col] - x_o[col])
            _clip_to_bounds(x_c, lower, upper)
            f_c = _ces_objective_from_free(
                x_c, fixed_mask, fixed_values, y, m, season, init_states
            )
            if f_c <= f_r:
                for col in range(n):
                    simplex[worst_idx, col] = x_c[col]
                f_simplex[worst_idx] = f_c
                continue
        else:
            for col in range(n):
                x_c[col] = x_o[col] - rho * (x_r[col] - x_o[col])
            _clip_to_bounds(x_c, lower, upper)
            f_c = _ces_objective_from_free(
                x_c, fixed_mask, fixed_values, y, m, season, init_states
            )
            if f_c < f_simplex[worst_idx]:
                for col in range(n):
                    simplex[worst_idx, col] = x_c[col]
                f_simplex[worst_idx] = f_c
                continue

        for pos in range(1, n + 1):
            row = order[pos]
            for col in range(n):
                simplex[row, col] = simplex[best_idx, col] + sigma * (
                    simplex[row, col] - simplex[best_idx, col]
                )
            _clip_to_bounds(simplex[row], lower, upper)
            f_simplex[row] = _ces_objective_from_free(
                simplex[row], fixed_mask, fixed_values, y, m, season, init_states
            )

    order = np.argsort(f_simplex)
    best_idx = order[0]
    best = np.empty(n, dtype=np.float64)
    for col in range(n):
        best[col] = simplex[best_idx, col]
    return best, n_iter, converged


def _seasonal_decompose_additive(y, period):
    """Classical additive seasonal factors used for CES ``P`` / ``F`` seeds."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    if period < 2 or n < 2 * period:
        return np.zeros(n, dtype=np.float64)

    if period % 2 == 0:
        filt = np.empty(period + 1, dtype=np.float64)
        filt[0] = 0.5 / period
        filt[-1] = 0.5 / period
        filt[1:-1] = 1.0 / period
    else:
        filt = np.full(period, 1.0 / period, dtype=np.float64)

    valid = np.convolve(y, filt, mode="valid")
    trend = np.full(n, np.nan, dtype=np.float64)
    trim_head = int(math.ceil(filt.shape[0] / 2.0) - 1)
    trim_tail = int(math.ceil(filt.shape[0] / 2.0) - (filt.shape[0] % 2))
    end = n - trim_tail if trim_tail else n
    trend[trim_head:end] = valid

    detrended = y - trend
    period_averages = np.empty(period, dtype=np.float64)
    for pos in range(period):
        period_averages[pos] = float(np.nanmean(detrended[pos::period]))
    period_averages -= float(np.nanmean(period_averages))
    seasonal = np.tile(period_averages, n // period + 1)[:n]
    seasonal[~np.isfinite(seasonal)] = 0.0
    return seasonal


@njit(cache=True, fastmath=True)
def _ces_yhat_from_states(states, i, m, season):
    """One-step fitted value from a CES state array."""
    if season == _CES_NONE or season == _CES_PARTIAL or season == _CES_FULL:
        yhat = states[i - 1, 0]
    else:
        yhat = states[i - m, 0]
    if season > _CES_SIMPLE:
        yhat += states[i - m, 2]
    return yhat


@njit(cache=True, fastmath=True)
def _ces_update_state(
    states,
    i,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
    y_value,
):
    """Update a CES state row, matching StatsForecast's ``cesupdate``."""
    if season == _CES_NONE or season == _CES_PARTIAL or season == _CES_FULL:
        eps = y_value - states[i - 1, 0]
    else:
        eps = y_value - states[i - m, 0]
    if season > _CES_SIMPLE:
        eps -= states[i - m, 2]

    if season == _CES_NONE or season == _CES_PARTIAL or season == _CES_FULL:
        states[i, 0] = (
            states[i - 1, 0]
            - (1.0 - alpha_1) * states[i - 1, 1]
            + (alpha_0 - alpha_1) * eps
        )
        states[i, 1] = (
            states[i - 1, 0]
            + (1.0 - alpha_0) * states[i - 1, 1]
            + (alpha_0 + alpha_1) * eps
        )
    else:
        states[i, 0] = (
            states[i - m, 0]
            - (1.0 - alpha_1) * states[i - m, 1]
            + (alpha_0 - alpha_1) * eps
        )
        states[i, 1] = (
            states[i - m, 0]
            + (1.0 - alpha_0) * states[i - m, 1]
            + (alpha_0 + alpha_1) * eps
        )

    if season == _CES_PARTIAL:
        states[i, 2] = states[i - m, 2] + beta_0 * eps
    elif season == _CES_FULL:
        states[i, 2] = (
            states[i - m, 2]
            - (1.0 - beta_1) * states[i - m, 3]
            + (beta_0 - beta_1) * eps
        )
        states[i, 3] = (
            states[i - m, 2]
            + (1.0 - beta_0) * states[i - m, 3]
            + (beta_0 + beta_1) * eps
        )
    return eps


@njit(cache=True, fastmath=True)
def _ces_forecast_from_states(
    h,
    states,
    n,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
):
    """Forecast recursively from a fitted CES state array."""
    n_components = states.shape[1]
    new_states = np.zeros((m + h, n_components), dtype=np.float64)
    for r in range(m):
        for c in range(n_components):
            new_states[r, c] = states[n + r, c]
    forecast = np.empty(h, dtype=np.float64)
    for i_h in range(m, m + h):
        yhat = _ces_yhat_from_states(new_states, i_h, m, season)
        forecast[i_h - m] = yhat
        _ces_update_state(
            new_states,
            i_h,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            yhat,
        )
    return forecast


@njit(cache=True, fastmath=True)
def _ces_one_step_forecast_from_states(
    states,
    n,
    m,
    season,
):
    """Return the one-step recursive forecast from a fitted CES state array."""
    if season == _CES_NONE or season == _CES_PARTIAL or season == _CES_FULL:
        yhat = states[n + m - 1, 0]
    else:
        yhat = states[n, 0]
    if season > _CES_SIMPLE:
        yhat += states[n, 2]
    return yhat


@njit(cache=True, fastmath=True)
def _ces_write_future_states(
    states,
    i,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
):
    """Populate the final ``m`` rows with zero-error future states."""
    n_components = states.shape[1]
    new_states = np.zeros((2 * m, n_components), dtype=np.float64)
    for r in range(m):
        for c in range(n_components):
            new_states[r, c] = states[i - m + r, c]
    for i_h in range(m, 2 * m):
        yhat = _ces_yhat_from_states(new_states, i_h, m, season)
        _ces_update_state(
            new_states,
            i_h,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            yhat,
        )
    start = states.shape[0] - m
    for r in range(m):
        for c in range(n_components):
            states[start + r, c] = new_states[m + r, c]


@njit(cache=True, fastmath=True)
def _reverse_rows_inplace(values):
    """Reverse rows of a 2D array in place."""
    n_rows = values.shape[0]
    n_cols = values.shape[1]
    for i in range(n_rows // 2):
        j = n_rows - 1 - i
        for col in range(n_cols):
            tmp = values[i, col]
            values[i, col] = values[j, col]
            values[j, col] = tmp


@njit(cache=True, fastmath=True)
def _reverse_1d_inplace(values):
    """Reverse a 1D array in place."""
    n = values.shape[0]
    for i in range(n // 2):
        j = n - 1 - i
        tmp = values[i]
        values[i] = values[j]
        values[j] = tmp


@njit(cache=True, fastmath={"contract"})
def _ces_fit_pass(
    y,
    states,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
    fitted,
    residuals,
):
    """Run one forward CES pass over ``y``."""
    n = y.shape[0]
    sse = 0.0
    for i in range(m, n + m):
        yhat = _ces_yhat_from_states(states, i, m, season)
        fitted[i - m] = yhat
        eps = y[i - m] - yhat
        residuals[i - m] = eps
        sse += eps * eps
        _ces_update_state(
            states,
            i,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            y[i - m],
        )
        if not np.isfinite(sse):
            return np.inf
    _ces_write_future_states(states, n + m, m, season, alpha_0, alpha_1, beta_0, beta_1)
    return sse


@njit(cache=True, fastmath={"contract"})
def _ces_fit_pass_sse(
    y,
    states,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
):
    """Run one forward CES pass and return SSE without fitted arrays."""
    n = y.shape[0]
    sse = 0.0
    for i in range(m, n + m):
        yhat = _ces_yhat_from_states(states, i, m, season)
        eps = y[i - m] - yhat
        sse += eps * eps
        _ces_update_state(
            states,
            i,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            y[i - m],
        )
        if not np.isfinite(sse):
            return np.inf
    _ces_write_future_states(states, n + m, m, season, alpha_0, alpha_1, beta_0, beta_1)
    return sse


@njit(cache=True, fastmath={"contract"})
def _ces_fit_pass_sse_ordered(
    y,
    states,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
    reverse,
):
    """Run one CES SSE pass over ``y`` in forward or reverse order."""
    n = y.shape[0]
    sse = 0.0
    for i in range(m, n + m):
        y_idx = n + m - 1 - i if reverse else i - m
        yhat = _ces_yhat_from_states(states, i, m, season)
        eps = y[y_idx] - yhat
        sse += eps * eps
        _ces_update_state(
            states,
            i,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            y[y_idx],
        )
        if not np.isfinite(sse):
            return np.inf
    _ces_write_future_states(states, n + m, m, season, alpha_0, alpha_1, beta_0, beta_1)
    return sse


@njit(cache=True, fastmath={"contract"})
def _ces_n_pass_sse_future(y, reverse, alpha_0, alpha_1, l1, l2):
    """Run one non-seasonal CES pass and return SSE plus future state."""
    n = y.shape[0]
    f12 = alpha_1 - 1.0
    f22 = 1.0 - alpha_0
    g1 = alpha_0 - alpha_1
    g2 = alpha_0 + alpha_1
    sse = 0.0
    for t in range(n):
        idx = n - 1 - t if reverse else t
        yhat = l1
        eps = y[idx] - yhat
        sse += eps * eps
        new_l1 = l1 + f12 * l2 + g1 * eps
        new_l2 = l1 + f22 * l2 + g2 * eps
        l1 = new_l1
        l2 = new_l2
        if not np.isfinite(sse):
            return np.inf, l1, l2

    new_l1 = l1 + f12 * l2
    new_l2 = l1 + f22 * l2
    return sse, new_l1, new_l2


@njit(cache=True, fastmath={"contract"})
def _ces_n_fit_objective_numba(y, alpha_0, alpha_1, init_real, init_imag):
    """Compute the non-seasonal CES backfit objective without state arrays."""
    n = y.shape[0]
    _, l1, l2 = _ces_n_pass_sse_future(y, False, alpha_0, alpha_1, init_real, init_imag)
    _, l1, l2 = _ces_n_pass_sse_future(y, True, alpha_0, alpha_1, l1, l2)
    sse, _, _ = _ces_n_pass_sse_future(y, False, alpha_0, alpha_1, l1, l2)

    if sse <= 0.0:
        return -np.inf
    if not np.isfinite(sse):
        return np.inf
    return n * math.log(sse)


@njit(cache=True, fastmath={"contract"})
def _ces_fit_objective_numba(
    y,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
    init_states,
):
    """Compute the StatsForecast-style CES backfit objective."""
    n = y.shape[0]
    n_components = init_states.shape[1]
    states = np.zeros((n + 2 * m, n_components), dtype=np.float64)
    for r in range(m):
        for c in range(n_components):
            states[r, c] = init_states[r, c]

    sse = _ces_fit_pass_sse_ordered(
        y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, False
    )

    _reverse_rows_inplace(states)
    sse = _ces_fit_pass_sse_ordered(
        y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, True
    )

    _reverse_rows_inplace(states)
    sse = _ces_fit_pass_sse_ordered(
        y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, False
    )

    if sse <= 0.0:
        return -np.inf
    if not np.isfinite(sse):
        return np.inf
    return n * math.log(sse)


@njit(cache=True, fastmath={"contract"})
def _ces_fit_states(
    y,
    m,
    season,
    alpha_0,
    alpha_1,
    beta_0,
    beta_1,
    init_states,
    backfit,
):
    """Fit CES states and return fitted values, residuals, states and objective."""
    n = y.shape[0]
    n_components = init_states.shape[1]
    states = np.zeros((n + 2 * m, n_components), dtype=np.float64)
    for r in range(m):
        for c in range(n_components):
            states[r, c] = init_states[r, c]

    y_work = y.copy()
    fitted = np.empty(n, dtype=np.float64)
    residuals = np.empty(n, dtype=np.float64)
    sse = _ces_fit_pass(
        y_work,
        states,
        m,
        season,
        alpha_0,
        alpha_1,
        beta_0,
        beta_1,
        fitted,
        residuals,
    )

    if backfit:
        _reverse_1d_inplace(y_work)
        _reverse_rows_inplace(states)
        _reverse_1d_inplace(residuals)
        sse = _ces_fit_pass(
            y_work,
            states,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            fitted,
            residuals,
        )

        _reverse_1d_inplace(y_work)
        _reverse_rows_inplace(states)
        _reverse_1d_inplace(residuals)
        sse = _ces_fit_pass(
            y_work,
            states,
            m,
            season,
            alpha_0,
            alpha_1,
            beta_0,
            beta_1,
            fitted,
            residuals,
        )

    if sse <= 0.0:
        likelihood_objective = -np.inf
    elif not np.isfinite(sse):
        likelihood_objective = np.inf
    else:
        likelihood_objective = n * math.log(sse)
    return fitted, residuals, states, likelihood_objective
