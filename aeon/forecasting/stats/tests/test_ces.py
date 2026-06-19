"""Tests for Complex Exponential Smoothing (CES) forecaster.

Phase 1 tests cover the non-seasonal ``model="N"`` recurrence; Phase 2
tests cover the seasonal ``model="S"``, ``model="P"``, and ``model="F"``
recurrences and the :class:`AutoCES` selector.
"""

import numpy as np
import pytest

from aeon.forecasting.stats import CES, AutoCES
from aeon.forecasting.stats._ces import (
    _CES_FULL,
    _CES_NONE,
    _ces_fit_states,
)

Y_EXAMPLE = np.array([2.1, 2.4, 2.8, 3.0, 3.6, 4.1, 4.4, 4.9, 5.3, 5.9, 6.2, 6.8])


def test_ces_fit_sets_attributes():
    """``fit`` should populate all documented attributes."""
    forecaster = CES().fit(Y_EXAMPLE)

    assert 0.01 <= forecaster.alpha_real_ <= 1.8
    assert 0.01 <= forecaster.alpha_imag_ <= 1.9
    assert np.isfinite(forecaster.initial_level_)
    assert np.isfinite(forecaster.level_real_)
    assert np.isfinite(forecaster.level_imag_)
    assert np.isfinite(forecaster.forecast_)
    assert np.isfinite(forecaster.sse_)
    assert forecaster.fitted_values_.shape == Y_EXAMPLE.shape
    assert forecaster.residuals_.shape == Y_EXAMPLE.shape
    # complex_alpha_ should agree with the two real-valued components.
    assert forecaster.complex_alpha_ == pytest.approx(
        forecaster.alpha_real_ + 1j * forecaster.alpha_imag_
    )


def test_ces_iterative_forecast_shape_and_finite():
    """``iterative_forecast`` returns the correct shape and finite values."""
    h = 5
    pred = CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=h)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (h,)
    assert np.all(np.isfinite(pred))


def test_ces_forecast_matches_iterative_h1():
    """``forecast_`` (stored after fit) equals ``iterative_forecast(y, 1)[0]``."""
    forecaster = CES().fit(Y_EXAMPLE)
    stored = forecaster.forecast_
    iterative = forecaster.iterative_forecast(Y_EXAMPLE, prediction_horizon=1)[0]
    assert np.isclose(stored, iterative)


def test_ces_predict_matches_iterative_h1():
    """``predict(y)`` after fit equals ``iterative_forecast(y, 1)[0]``."""
    forecaster = CES().fit(Y_EXAMPLE)
    predicted = forecaster.predict(Y_EXAMPLE)
    iterative = forecaster.iterative_forecast(Y_EXAMPLE, prediction_horizon=1)[0]
    assert np.isclose(predicted, iterative)


def test_ces_fixed_parameters_are_honoured():
    """Parameters fixed by the user should appear unchanged in fitted attrs."""
    forecaster = CES(
        alpha_real=0.5,
        alpha_imag=0.2,
        initial_level=2.0,
    ).fit(Y_EXAMPLE)
    assert forecaster.alpha_real_ == pytest.approx(0.5)
    assert forecaster.alpha_imag_ == pytest.approx(0.2)
    assert forecaster.initial_level_ == pytest.approx(2.0)


def test_ces_optimiser_reduces_objective():
    """The optimised fit should beat the user-fixed baseline on SSE."""
    fixed = CES(
        alpha_real=0.5,
        alpha_imag=0.01,
        initial_level=float(Y_EXAMPLE[0]),
    ).fit(Y_EXAMPLE)
    optimised = CES().fit(Y_EXAMPLE)
    assert optimised.sse_ <= fixed.sse_ + 1e-9


def test_ces_constant_series_returns_finite_forecasts():
    """Constant series must fit and forecast without numerical failures."""
    y = np.full(20, 4.0)
    pred = CES().iterative_forecast(y, prediction_horizon=4)
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_ces_constant_series_with_admissible_alpha_recovers_constant():
    """With ``alpha_imag = 1`` the constant-series equilibrium is preserved.

    Setting ``alpha_imag = 1`` zeroes the off-diagonal entry of the transition
    matrix that mixes the imaginary state into the observed level, so a
    constant series at ``c`` fitted with ``initial_level = c`` produces
    constant ``c`` forecasts. This sanity-checks the recurrence rather than
    the optimiser.
    """
    y = np.full(20, 4.0)
    pred = CES(
        alpha_real=0.5,
        alpha_imag=1.0,
        initial_level=4.0,
    ).iterative_forecast(y, prediction_horizon=4)
    np.testing.assert_allclose(pred, 4.0, atol=1e-9)


def test_ces_short_series_raises():
    """CES should require at least two observations."""
    with pytest.raises(ValueError, match="at least 2 observations"):
        CES().fit(np.array([1.0]))


def test_ces_non_finite_series_raises():
    """Non-finite values in ``y`` should be rejected.

    The base class catches missing values before the CES-level finite check,
    so the message can come from either layer. Any ValueError is acceptable.
    """
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.raises(ValueError):
        CES().fit(y)


def test_ces_exog_raises_not_implemented():
    """Exogenous variables are not supported in Phase 1."""
    exog = np.arange(Y_EXAMPLE.shape[0], dtype=float)
    with pytest.raises(NotImplementedError, match="exogenous"):
        CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=2, exog=exog)


def test_ces_future_exog_raises_not_implemented():
    """``future_exog`` is also rejected in Phase 1."""
    future = np.arange(3, dtype=float)
    with pytest.raises(NotImplementedError, match="exogenous"):
        CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=3, future_exog=future)


def test_ces_invalid_horizon_raises():
    """A non-positive prediction horizon must raise."""
    with pytest.raises(ValueError, match="prediction_horizon"):
        CES().iterative_forecast(Y_EXAMPLE, prediction_horizon=0)


def test_ces_invalid_bounds_raise():
    """Invalid bounds must raise during fit."""
    with pytest.raises(ValueError, match="alpha_real_bounds"):
        CES(alpha_real_bounds=(1.0, 0.0)).fit(Y_EXAMPLE)


def test_ces_fixed_out_of_bounds_raises():
    """A fixed value outside its bounds must raise during fit."""
    with pytest.raises(ValueError, match="alpha_real"):
        CES(
            alpha_real=2.5,
            alpha_real_bounds=(0.01, 1.8),
        ).fit(Y_EXAMPLE)


# ---------------------------------------------------------------------------
# Phase 2: hand-calculated recurrence checks
# ---------------------------------------------------------------------------


def test_ces_recursion_hand_calc_non_seasonal():
    """Hand-trace the non-seasonal recurrence with the smooth/ADAM form.

    With ``(alpha_0, alpha_1) = (1, 0)`` and zero initial state, the
    rotated persistence ``g = (alpha_0 - alpha_1, alpha_0 + alpha_1) = (1, 1)``
    combined with ``F = [[1, -1], [1, 0]]`` gives a clean trace on
    ``y = [1, 2, 3]``:

    ``t=0``: yhat=0, eps=1, new state (1, 1).
    ``t=1``: yhat=1, eps=1, new state (1+(-1)*1+1, 1+0*1+1) = (1, 2).
    ``t=2``: yhat=1, eps=2, new state (1+(-1)*2+2, 1+0*2+2) = (1, 3).
    """
    y = np.array([1.0, 2.0, 3.0])
    init_states = np.zeros((1, 2), dtype=np.float64)
    fitted, residuals, states, _ = _ces_fit_states(
        y,
        1,
        _CES_NONE,
        1.0,
        0.0,
        np.nan,
        np.nan,
        init_states,
        False,
    )
    np.testing.assert_allclose(fitted, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(residuals, [1.0, 1.0, 2.0])
    assert states[y.shape[0], 0] == pytest.approx(1.0)
    assert states[y.shape[0], 1] == pytest.approx(3.0)


def test_ces_recursion_hand_calc_full_seasonal():
    """Hand-trace the full seasonal recurrence with the smooth/ADAM form.

    With ``alpha = (1, 0)``, ``beta = (1, 0)`` both the non-seasonal and
    the seasonal pair use persistence ``g = (1, 1)`` and transition rows
    ``[[1, -1], [1, 0]]``. Stepping through ``y = 1..8`` with ``m = 4`` and
    zero initial state for everything gives:

    * ``t=0..3``: only the level fires; ``yhat = l0 + l1[idx] = l0`` and
      the level chases ``y`` with lag 1, depositing ``l1[idx] = t+1`` at
      each step.
    * ``t=4``: ``yhat = 1 + 1 = 2``, ``eps = 3``.
    * ``t=5``: ``yhat = 0 + 1 = 1``, ``eps = 5``.
    * ``t=6``: ``yhat = 1 + 2 = 3``, ``eps = 4``.
    * ``t=7``: ``yhat = 0 + 3 = 3``, ``eps = 5``.

    Net fitted sequence: ``[0, 1, 1, 1, 2, 1, 3, 3]``.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    init_states = np.zeros((4, 4), dtype=np.float64)
    fitted, residuals, states, _ = _ces_fit_states(
        y,
        4,
        _CES_FULL,
        1.0,
        0.0,
        1.0,
        0.0,
        init_states,
        False,
    )
    expected_fitted = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 3.0])
    np.testing.assert_allclose(fitted, expected_fitted)
    np.testing.assert_allclose(residuals, y - expected_fitted)
    final_buffer = states[y.shape[0] : y.shape[0] + 4]
    assert np.all(np.isfinite(final_buffer[:, :2]))
    assert final_buffer[:, 2].shape == (4,)
    assert final_buffer[:, 3].shape == (4,)


# ---------------------------------------------------------------------------
# Phase 2: full seasonal model ("F")
# ---------------------------------------------------------------------------


def _synthetic_seasonal(seed=0):
    """Trend + clean additive seasonality + small noise, ``m=4``."""
    rng = np.random.default_rng(seed)
    n = 40
    trend = np.linspace(10.0, 30.0, n)
    season = np.array([2.0, -1.0, -3.0, 2.0])
    return trend + np.tile(season, n // 4) + rng.normal(0.0, 0.3, n)


def test_ces_full_fits_seasonal_series():
    """CES with model='F' fits a clean seasonal series and stores all attrs."""
    y = _synthetic_seasonal()
    f = CES(model="F", season_length=4).fit(y)
    assert f.model_ == "F"
    assert f.season_length_ == 4
    assert 0.01 <= f.alpha_real_ <= 1.8
    assert 0.01 <= f.beta_real_ <= 1.5
    assert f.seasonal_real_.shape == (4,)
    assert f.seasonal_imag_.shape == (4,)
    assert f.fitted_values_.shape == y.shape
    assert f.residuals_.shape == y.shape
    assert np.isfinite(f.forecast_)
    assert np.isfinite(f.sse_)
    assert f.n_params_ == 5


def test_ces_full_iterative_forecast_shape_and_finite():
    """Seasonal ``iterative_forecast`` returns ``(h,)`` finite values."""
    y = _synthetic_seasonal()
    h = 8
    pred = CES(model="F", season_length=4).iterative_forecast(y, prediction_horizon=h)
    assert pred.shape == (h,)
    assert np.all(np.isfinite(pred))


def test_ces_full_predict_matches_iterative_h1():
    """``forecast_``, ``predict(y)``, and ``iterative_forecast(y,1)[0]`` agree."""
    y = _synthetic_seasonal()
    f = CES(model="F", season_length=4).fit(y)
    stored = f.forecast_
    predicted = f.predict(y)
    iterative = f.iterative_forecast(y, prediction_horizon=1)[0]
    assert np.isclose(stored, predicted)
    assert np.isclose(stored, iterative)


def test_ces_full_fixed_seasonal_parameters_honoured():
    """Fixed seasonal smoothing parameters appear unchanged in fitted attrs."""
    y = _synthetic_seasonal()
    f = CES(
        model="F",
        season_length=4,
        alpha_real=0.5,
        alpha_imag=0.5,
        beta_real=0.3,
        beta_imag=0.4,
    ).fit(y)
    assert f.alpha_real_ == pytest.approx(0.5)
    assert f.alpha_imag_ == pytest.approx(0.5)
    assert f.beta_real_ == pytest.approx(0.3)
    assert f.beta_imag_ == pytest.approx(0.4)


def test_ces_full_optimised_sse_no_worse_than_fixed():
    """The optimised seasonal fit must match or beat a fixed baseline."""
    y = _synthetic_seasonal()
    fixed = CES(
        model="F",
        season_length=4,
        alpha_real=0.5,
        alpha_imag=0.5,
        beta_real=0.5,
        beta_imag=0.5,
    ).fit(y)
    optimised = CES(model="F", season_length=4).fit(y)
    assert optimised.sse_ <= fixed.sse_ + 1e-9


def test_ces_full_rejects_short_series():
    """Seasonal fit must reject ``n < 2 * season_length``."""
    y = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        CES(model="F", season_length=4).fit(y)


def test_ces_rejects_season_length_one_for_seasonal_model():
    """Seasonal model requires ``season_length >= 2``."""
    with pytest.raises(ValueError, match="season_length"):
        CES(model="F", season_length=1).fit(Y_EXAMPLE)


def test_ces_invalid_model_raises():
    """Unknown model codes raise ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown CES model"):
        CES(model="Q").fit(Y_EXAMPLE)


@pytest.mark.parametrize("model", ["S", "P", "simple", "partial"])
def test_ces_simple_and_partial_fit(model):
    """``S`` and ``P`` seasonal recurrences fit and forecast finite values."""
    y = _synthetic_seasonal()
    f = CES(model=model, season_length=4).fit(y)
    assert f.model_ in ("S", "P")
    assert f.season_length_ == 4
    assert np.isfinite(f.forecast_)
    assert np.all(np.isfinite(f.fitted_values_))
    assert f.iterative_forecast(y, prediction_horizon=4).shape == (4,)


@pytest.mark.parametrize(
    "alias,expected",
    [("none", "N"), ("simple", "S"), ("partial", "P"), ("full", "F")],
)
def test_ces_long_form_aliases(alias, expected):
    """Long-form aliases resolve to their canonical model codes."""
    y = _synthetic_seasonal()
    f = CES(model=alias, season_length=4 if expected != "N" else 1).fit(y)
    assert f.model_ == expected


@pytest.mark.parametrize("model", ["N", "S", "P", "F"])
def test_ces_matches_statsforecast_reference_forecasts(model):
    """Aeon CES forecasts should broadly match StatsForecast for each model."""
    statsforecast_models = pytest.importorskip("statsforecast.models")
    y = np.arange(1, 41, dtype=float) + np.tile([2.0, -1.0, -3.0, 2.0], 10)
    h = 6
    season_length = 1 if model == "N" else 4

    aeon_pred = CES(model=model, season_length=season_length).iterative_forecast(
        y, prediction_horizon=h
    )

    statsforecast_model = statsforecast_models.AutoCES(
        season_length=season_length, model=model
    )
    statsforecast_model.fit(y)
    statsforecast_pred = statsforecast_model.predict(h=h)["mean"]

    np.testing.assert_allclose(aeon_pred, statsforecast_pred, rtol=1e-4, atol=5e-2)


def test_autoces_matches_statsforecast_auto_reference():
    """Aeon AutoCES should select the same model and forecasts as SF AutoCES."""
    statsforecast_models = pytest.importorskip("statsforecast.models")
    y = np.arange(1, 41, dtype=float) + np.tile([2.0, -1.0, -3.0, 2.0], 10)
    h = 6

    aeon_auto = AutoCES(season_length=4)
    aeon_pred = aeon_auto.iterative_forecast(y, prediction_horizon=h)

    statsforecast_auto = statsforecast_models.AutoCES(season_length=4, model="Z")
    statsforecast_auto.fit(y)
    statsforecast_pred = statsforecast_auto.predict(h=h)["mean"]
    statsforecast_selected = statsforecast_auto.model_["seasontype"]

    assert aeon_auto.best_model_name_ == statsforecast_selected
    np.testing.assert_allclose(aeon_pred, statsforecast_pred, rtol=1e-4, atol=5e-2)


# ---------------------------------------------------------------------------
# AutoCES
# ---------------------------------------------------------------------------


def test_autoces_constant_series_prefers_non_seasonal():
    """On a flat series ``AutoCES`` should pick ``"N"`` over seasonal models."""
    y = np.full(30, 4.0)
    auto = AutoCES(season_length=4).fit(y)
    assert auto.best_model_name_ == "N"


def test_autoces_seasonal_series_tries_all_models():
    """On seasonal data ``AutoCES`` evaluates the full StatsForecast search set."""
    y = _synthetic_seasonal()
    auto = AutoCES(season_length=4).fit(y)
    assert auto.best_model_name_ in {"S", "P", "F"}
    assert set(auto.model_results_.keys()) == {"N", "S", "P", "F"}
    for code in ("N", "S", "P", "F"):
        assert auto.model_results_[code]["status"] == "ok"
    assert auto.model_results_[auto.best_model_name_]["aicc"] == min(
        result["aicc"] for result in auto.model_results_.values()
    )


def test_autoces_records_candidate_results_cleanly():
    """``AutoCES`` records successful IC results for every viable candidate."""
    y = _synthetic_seasonal()
    auto = AutoCES(season_length=4, models=("N", "S", "P", "F")).fit(y)
    for code in ("N", "S", "P", "F"):
        assert auto.model_results_[code]["status"] == "ok"
        assert np.isfinite(auto.model_results_[code]["sse"])
    assert auto.best_model_name_ in ("N", "S", "P", "F")


def test_autoces_skips_seasonal_when_season_length_is_one():
    """With ``season_length=1`` seasonal candidates are skipped, ``N`` wins."""
    y = _synthetic_seasonal()
    auto = AutoCES(season_length=1, models=("N", "F")).fit(y)
    assert auto.best_model_name_ == "N"
    assert auto.model_results_["F"]["status"] == "skipped"


def test_autoces_iterative_forecast_passes_through():
    """``AutoCES.iterative_forecast`` returns the selected model's forecasts."""
    y = _synthetic_seasonal()
    h = 6
    auto = AutoCES(season_length=4)
    auto_pred = auto.iterative_forecast(y, prediction_horizon=h)
    direct = auto.best_model_.iterative_forecast(y, prediction_horizon=h)
    np.testing.assert_allclose(auto_pred, direct)


def test_autoces_invalid_ic_raises():
    """An unknown information criterion must raise ``ValueError``."""
    y = _synthetic_seasonal()
    with pytest.raises(ValueError, match="ic must be one of"):
        AutoCES(season_length=4, ic="nonsense").fit(y)


@pytest.mark.parametrize("bad", [0, -1, 4.9, True, False, "4"])
def test_autoces_invalid_season_length_raises(bad):
    """AutoCES must reject the same bad season_length values as CES.

    Regression for a Codex review point: ``AutoCES._fit`` previously
    called ``int(self.season_length)`` without validation, so floats,
    strings, booleans and non-positive ints all silently coerced into a
    valid-looking ``m`` and the seasonal candidates were quietly skipped.
    """
    y = _synthetic_seasonal()
    with pytest.raises(ValueError, match="season_length"):
        AutoCES(season_length=bad).fit(y)


def test_autoces_all_candidates_failing_raises():
    """If every candidate is skipped or errors, fitting must raise."""
    y = _synthetic_seasonal()
    # season_length=1 → F skipped; only "S" requested → also skipped → all fail
    with pytest.raises(ValueError, match="All AutoCES candidates failed"):
        AutoCES(season_length=1, models=("S",)).fit(y)


def test_autoces_rejects_exog_and_future_exog():
    """AutoCES rejects exogenous inputs at the iterative_forecast layer."""
    y = _synthetic_seasonal()
    exog = np.arange(y.shape[0], dtype=float)
    with pytest.raises(NotImplementedError, match="exogenous"):
        AutoCES(season_length=4).iterative_forecast(y, prediction_horizon=3, exog=exog)
    with pytest.raises(NotImplementedError, match="exogenous"):
        AutoCES(season_length=4).iterative_forecast(
            y, prediction_horizon=3, future_exog=np.arange(3.0)
        )
