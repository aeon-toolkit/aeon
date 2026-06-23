# Run:
#   python experiments/forecasting/m_comp_experiments.py
#   python experiments/forecasting/m_comp_experiments.py --datasets M1 \
#       --forecasters ETS DOTM ARIMA --backends aeon statsforecast
#   python experiments/forecasting/m_comp_experiments.py --datasets M4 \
#       --m4-home E:\Data\m4 --types yearly --forecasters ETS
#   python experiments/forecasting/m_comp_experiments.py --datasets M4 \
#       --limit-per-type 0 --forecasters ETS DOTM ARIMA CES \
#       --backends aeon statsforecast --n-jobs 8

"""Persist M-competition forecaster outputs for later analysis.

This script creates reusable result files so ensembles and comparisons can be
built without refitting base forecasters. For each requested backend, competition,
and forecaster it writes two files:

* ``<backend>/<forecaster>/<competition>_<forecaster>_Test.csv`` with h-step
  test forecasts and test metrics.
* ``<backend>/<forecaster>/<competition>_<forecaster>_Train.csv`` with in-sample
  fitted values, residuals, training metrics, selected model details, and fit
  statistics.

The files have a small metadata header followed by ``@results`` and a CSV table.
They are intended as local investigation artifacts, not package test fixtures.

Examples
--------
Run aeon and statsforecast equivalents over the default M1/M3 sample:

    python experiments/forecasting/m_comp_experiments.py \
        --forecasters ETS DOTM ARIMA \
        --backends aeon statsforecast

Smoke test on one M1 series type:

    python experiments/forecasting/m_comp_experiments.py --datasets M1 \
        --limit-per-type 1 --forecasters ETS DOTM ARIMA \
        --backends aeon statsforecast --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

DEFAULT_DATA_HOME = Path(
    os.environ.get("AEON_FCOMPDATA_HOME", ROOT / "data" / "fcompdata")
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "AEON_MCOMP_OUTPUT_DIR",
        r"D:\Results\Forecasting\M Competitions",
    )
)
DEFAULT_M4_HOME = Path(os.environ.get("AEON_M4_HOME", r"D:\Data\Forecasting\m4"))
DEFAULT_BACKENDS = ("aeon",)
DEFAULT_FORECASTERS = ("AutoETS", "DOTM")
DEFAULT_N_JOBS = min(32, (os.cpu_count() or 1) + 4)
SERIES_BATCH_SIZE = 1000
BACKEND_ALIASES = {
    "aeon": "aeon",
    "statsforecast": "statsforecast",
    "sf": "statsforecast",
}
M4_FILES = {
    "yearly": ("m4_yearly_dataset.tsf", 1, 1),
    "quarterly": ("m4_quarterly_dataset.tsf", 23001, 4),
    "monthly": ("m4_monthly_dataset.tsf", 47001, 12),
    "weekly": ("m4_weekly_dataset.tsf", 95001, 1),
    "daily": ("m4_daily_dataset.tsf", 95360, 1),
    "hourly": ("m4_hourly_dataset.tsf", 99587, 24),
}
FORECASTER_ALIASES = {
    "ARIMA": "ARIMA",
    "AutoARIMA": "AutoARIMA",
    "AutoCES": "AutoCES",
    "AutoETS": "AutoETS",
    "AutoTAR": "AutoTAR",
    "DrCIF": "DrCIF",
    "ETS": "ETS",
    "CES": "CES",
    "DOTM": "DOTM",
    "DeepARForecaster": "DeepARForecaster",
    "NBeatsForecaster": "NBeatsForecaster",
    "NaiveForecaster": "NaiveForecaster",
    "RegressionForecaster": "RegressionForecaster",
    "SETAR": "SETAR",
    "SETARForest": "SETARForest",
    "SETARTree": "SETARTree",
    "TAR": "TAR",
    "TCNForecaster": "TCNForecaster",
    "TVP": "TVP",
    "Theta": "Theta",
}


@dataclass
class FitResult:
    """Container for one fitted forecaster and its reusable outputs."""

    forecast: np.ndarray | None
    fitted_values: np.ndarray | None
    residuals: np.ndarray | None
    train_seconds: float
    elapsed_seconds: float
    status: str
    model: str
    params: dict[str, Any]
    fit_stats: dict[str, Any]
    error: str = ""


@dataclass
class SelectedSeries:
    """A selected M-competition series and its derived arrays."""

    dataset: str
    index: int
    series: str
    series_type: str
    period: int
    y_train: np.ndarray
    y_test: np.ndarray
    h: int

    @property
    def key(self) -> str:
        """Return a stable dataset:index key."""
        return f"{self.dataset}:{self.index}"


def _patch_fcompdata_home(data_home: Path) -> None:
    """Point fcompdata at a local cache directory."""
    import fcompdata.download as download

    data_home.mkdir(parents=True, exist_ok=True)

    def get_data_home() -> Path:
        return data_home

    download.get_data_home = get_data_home


def _dataset(name: str):
    """Return an fcompdata dataset object by name."""
    import fcompdata

    datasets = {
        "M1": fcompdata.M1,
        "M3": fcompdata.M3,
        "M4": fcompdata.M4,
        "Tourism": fcompdata.Tourism,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset {name!r}")
    return datasets[name]


def _as_float_array(values) -> np.ndarray:
    """Return a 1D float64 array."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute symmetric mean absolute percentage error."""
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    if not np.any(valid):
        return math.nan
    values = 200.0 * np.abs(y_true[valid] - y_pred[valid]) / denom[valid]
    return float(np.mean(values))


def _mase(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    period: int,
) -> float:
    """Compute mean absolute scaled error."""
    lag = max(1, int(period))
    if y_train.size <= lag:
        return math.nan
    scale = float(np.mean(np.abs(y_train[lag:] - y_train[:-lag])))
    if not np.isfinite(scale) or scale <= 0:
        return math.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _metrics(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray | None,
    period: int,
) -> dict[str, float | bool]:
    """Return forecast-error metrics for one train/test split."""
    if (
        y_pred is None
        or y_pred.shape != y_true.shape
        or not np.all(np.isfinite(y_pred))
    ):
        return {
            "finite": False,
            "smape": math.nan,
            "mase": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
        }
    error = y_true - y_pred
    return {
        "finite": True,
        "smape": _smape(y_true, y_pred),
        "mase": _mase(y_train, y_true, y_pred, period),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def _selected_from_raw(dataset_name: str, index: int, raw) -> SelectedSeries:
    """Convert an fcompdata row into the local selected-series container."""
    y_train = _as_float_array(raw.x)
    y_test = _as_float_array(raw.xx)
    h = int(raw.h)
    if y_test.size != h:
        y_test = y_test[:h]
    return SelectedSeries(
        dataset=dataset_name,
        index=int(index),
        series=str(raw.sn),
        series_type=str(raw.type),
        period=max(1, int(raw.period)),
        y_train=y_train,
        y_test=y_test,
        h=h,
    )


def _json_safe(value):
    """Convert arrays/scalars into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _json_dumps(value) -> str:
    """Serialise metadata compactly."""
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _encode_vector(values: np.ndarray | None) -> str:
    """Pipe-encode a numeric vector for variable-length train outputs."""
    if values is None:
        return ""
    return "|".join(
        "" if not math.isfinite(float(v)) else f"{float(v):.17g}" for v in values
    )


def _blankable(value) -> str | float:
    """Return blank for non-finite values, otherwise the original float."""
    value = float(value)
    return value if math.isfinite(value) else ""


def _train_metrics(y_train: np.ndarray, fitted: np.ndarray | None, period: int) -> dict:
    """Compute in-sample fitted-value metrics."""
    if fitted is None:
        return {"smape": math.nan, "mase": math.nan, "mae": math.nan, "rmse": math.nan}
    fitted = _as_float_array(fitted)
    if (
        fitted.size == 0
        or fitted.size > y_train.size
        or not np.all(np.isfinite(fitted))
    ):
        return {"smape": math.nan, "mase": math.nan, "mae": math.nan, "rmse": math.nan}

    y_eval = y_train[y_train.size - fitted.size :]
    error = y_eval - fitted
    denom = np.abs(y_eval) + np.abs(fitted)
    valid = denom > 0
    smape = math.nan
    if np.any(valid):
        smape = float(np.mean(200.0 * np.abs(error[valid]) / denom[valid]))

    mase = math.nan
    lag = max(1, int(period))
    if y_train.size > lag:
        scale = float(np.mean(np.abs(y_train[lag:] - y_train[:-lag])))
        if math.isfinite(scale) and scale > 0:
            mase = float(np.mean(np.abs(error)) / scale)

    return {
        "smape": smape,
        "mase": mase,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def _m4_tsf_rows(path: Path):
    """Yield ``(series_name, values, horizon, frequency)`` rows from an M4 TSF file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find M4 TSF file {path}. Set --m4-home or AEON_M4_HOME to "
            "the directory containing the M4 TSF files."
        )
    frequency = None
    horizon = None
    in_data = False
    with path.open("r", encoding="cp1252") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if not in_data:
                if lower.startswith("@frequency"):
                    frequency = line.split(maxsplit=1)[1].strip().lower()
                elif lower.startswith("@horizon"):
                    horizon = int(line.split(maxsplit=1)[1])
                elif lower == "@data":
                    if frequency is None or horizon is None:
                        raise ValueError(
                            f"Missing frequency/horizon metadata in {path}"
                        )
                    in_data = True
                continue

            fields = line.split(":", 2)
            if len(fields) != 3:
                raise ValueError(f"Cannot parse M4 TSF row in {path}: {line[:80]!r}")
            values = np.asarray(
                [float(value) for value in fields[2].split(",")],
                dtype=np.float64,
            )
            yield fields[0], values, horizon, frequency


def _m4_selected_from_values(
    *,
    index: int,
    series_name: str,
    series_type: str,
    period: int,
    values: np.ndarray,
    h: int,
) -> SelectedSeries:
    """Create a selected M4 series by holding out the TSF horizon."""
    if values.size <= h:
        raise ValueError(f"M4:{index} has too few observations for horizon {h}.")
    return SelectedSeries(
        dataset="M4",
        index=int(index),
        series=str(series_name),
        series_type=str(series_type),
        period=int(period),
        y_train=values[:-h],
        y_test=values[-h:],
        h=int(h),
    )


def _select_m4_series(args: argparse.Namespace) -> list[SelectedSeries]:
    """Select M4 series from local TSF files."""
    selected: list[SelectedSeries] = []
    wanted_types = set(args.types)
    for series_type, (filename, start_index, period) in M4_FILES.items():
        if wanted_types and series_type not in wanted_types:
            continue
        path = args.m4_home / filename
        limit = math.inf if args.limit_per_type <= 0 else int(args.limit_per_type)
        for offset, (series_name, values, h, frequency) in enumerate(
            _m4_tsf_rows(path)
        ):
            if offset >= limit:
                break
            selected.append(
                _m4_selected_from_values(
                    index=start_index + offset,
                    series_name=series_name,
                    series_type=frequency,
                    period=period,
                    values=values,
                    h=h,
                )
            )
    return selected


def _select_m4_index(index: int, m4_home: Path) -> SelectedSeries:
    """Select a single M4 series by global M4 index."""
    for _series_type, (filename, start_index, period) in M4_FILES.items():
        path = m4_home / filename
        target_offset = index - start_index
        if target_offset < 0:
            continue
        for offset, (series_name, values, h, frequency) in enumerate(
            _m4_tsf_rows(path)
        ):
            if offset == target_offset:
                return _m4_selected_from_values(
                    index=index,
                    series_name=series_name,
                    series_type=frequency,
                    period=period,
                    values=values,
                    h=h,
                )
            if offset > target_offset:
                break
    raise ValueError(f"Unknown M4 index {index}.")


def select_experiment_series(args: argparse.Namespace) -> list[SelectedSeries]:
    """Select requested M-competition series for persistence experiments."""
    selected: list[SelectedSeries] = []
    for dataset_name in args.datasets:
        if dataset_name == "M4":
            selected.extend(_select_m4_series(args))
            continue

        counts: dict[str, int] = {}
        dataset = _dataset(dataset_name)
        for index, raw in dataset.items():
            if args.types and raw.type not in args.types:
                continue
            current = counts.get(raw.type, 0)
            if args.limit_per_type > 0 and current >= args.limit_per_type:
                continue
            counts[raw.type] = current + 1
            selected.append(_selected_from_raw(dataset_name, index, raw))

    for token in args.include_indices:
        dataset_name, index_text = token.split(":", 1)
        index = int(index_text)
        if dataset_name == "M4":
            selected.append(_select_m4_index(index, args.m4_home))
        else:
            raw = _dataset(dataset_name)[index]
            selected.append(_selected_from_raw(dataset_name, index, raw))

    deduped: list[SelectedSeries] = []
    seen = set()
    for item in selected:
        if item.key not in seen:
            seen.add(item.key)
            deduped.append(item)
    return deduped


def _ets_forecast_from_fit(model, h: int) -> np.ndarray:
    """Forecast AutoETS from the fitted wrapped ETS state without refitting."""
    from aeon.forecasting.stats._ets import _numba_predict

    wrapped = model.wrapped_model_
    preds = np.empty(h, dtype=np.float64)
    for i in range(h):
        preds[i] = _numba_predict(
            wrapped._trend_type,
            wrapped._seasonality_type,
            wrapped.level_,
            wrapped.trend_,
            wrapped.seasonality_,
            wrapped.phi_,
            i + 1,
            wrapped.n_timepoints_,
            wrapped._seasonal_period,
        )
    return preds


def _fit_ets(y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit aeon AutoETS once and return reusable outputs."""
    from aeon.forecasting.stats import AutoETS

    start = time.perf_counter()
    model = AutoETS(seasonal_period=max(1, int(period))).fit(y)
    train_seconds = time.perf_counter() - start
    forecast = _ets_forecast_from_fit(model, h)
    elapsed = time.perf_counter() - start
    wrapped = model.wrapped_model_
    params = {
        "error_type": model.error_type_,
        "trend_type": model.trend_type_,
        "seasonality_type": model.seasonality_type_,
        "seasonal_period": model.seasonal_period_,
        "alpha": getattr(wrapped, "alpha_", math.nan),
        "beta": getattr(wrapped, "beta_", math.nan),
        "gamma": getattr(wrapped, "gamma_", math.nan),
        "phi": getattr(wrapped, "phi_", math.nan),
        "parameters": getattr(wrapped, "parameters_", []),
    }
    fit_stats = {
        "aic": getattr(wrapped, "aic_", math.nan),
        "avg_mean_sq_err": getattr(wrapped, "avg_mean_sq_err_", math.nan),
        "likelihood": getattr(wrapped, "likelihood_", math.nan),
        "k": getattr(wrapped, "k_", math.nan),
    }
    model_code = "".join(
        {0: "N", 1: "A", 2: "M"}.get(int(v), str(v))
        for v in (model.error_type_, model.trend_type_, model.seasonality_type_)
    )
    return FitResult(
        forecast=forecast,
        fitted_values=_as_float_array(wrapped.fitted_values_),
        residuals=_as_float_array(wrapped.residuals_),
        train_seconds=train_seconds,
        elapsed_seconds=elapsed,
        status="ok",
        model=model_code,
        params=params,
        fit_stats=fit_stats,
    )


def _ces_forecast_from_fit(model, h: int) -> np.ndarray:
    """Forecast AutoCES from the selected fitted model without refitting."""
    from aeon.forecasting.stats._ces import _ces_forecast_from_states

    best = model.best_model_
    return _ces_forecast_from_states(
        int(h),
        best._states_,
        int(best._n_train_),
        best._fit_m_,
        best._season_code_,
        best.alpha_real_,
        best.alpha_imag_,
        best.beta_real_,
        best.beta_imag_,
    )


def _fit_ces(y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit aeon AutoCES once and return reusable outputs."""
    from aeon.forecasting.stats import AutoCES

    start = time.perf_counter()
    model = AutoCES(season_length=max(1, int(period))).fit(y)
    train_seconds = time.perf_counter() - start
    forecast = _ces_forecast_from_fit(model, h)
    elapsed = time.perf_counter() - start
    best = model.best_model_
    params = {
        "best_model": model.best_model_name_,
        "season_length": getattr(best, "season_length_", math.nan),
        "alpha_real": getattr(best, "alpha_real_", math.nan),
        "alpha_imag": getattr(best, "alpha_imag_", math.nan),
        "beta_real": getattr(best, "beta_real_", math.nan),
        "beta_imag": getattr(best, "beta_imag_", math.nan),
        "initial_level": getattr(best, "initial_level_", math.nan),
        "initial_level_imag": getattr(best, "initial_level_imag_", math.nan),
    }
    fit_stats = {
        "sse": getattr(best, "sse_", math.nan),
        "n_params": getattr(best, "n_params_", math.nan),
        "n_free_params": getattr(best, "n_free_params_", math.nan),
        "n_iter": getattr(best, "n_iter_", math.nan),
        "optimization_success": getattr(best, "optimization_success_", None),
        "model_results": getattr(model, "model_results_", {}),
    }
    return FitResult(
        forecast=forecast,
        fitted_values=_as_float_array(best.fitted_values_),
        residuals=_as_float_array(best.residuals_),
        train_seconds=train_seconds,
        elapsed_seconds=elapsed,
        status="ok",
        model=str(model.best_model_name_),
        params=params,
        fit_stats=fit_stats,
    )


def _arima_original_scale_fitted(model, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute original-scale expanding one-step fitted values for AutoARIMA."""
    fitted = []
    fitted_start = None
    final_model = model.final_model_
    for t in range(1, y.size):
        try:
            pred = float(final_model.predict(y[:t]))
        except Exception:  # noqa: BLE001
            if fitted_start is not None:
                break
            continue
        if not math.isfinite(pred):
            if fitted_start is not None:
                break
            continue
        if fitted_start is None:
            fitted_start = t
        fitted.append(pred)

    if fitted_start is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    fitted_values = np.asarray(fitted, dtype=np.float64)
    actual = y[fitted_start : fitted_start + fitted_values.size]
    return fitted_values, actual - fitted_values


def _fit_arima(y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit aeon AutoARIMA once and return reusable outputs."""
    from aeon.forecasting.stats import AutoARIMA

    start = time.perf_counter()
    model = AutoARIMA().fit(y)
    train_seconds = time.perf_counter() - start
    forecast = _as_float_array(
        model.final_model_._iterative_forecast_from_fitted(prediction_horizon=int(h))
    )
    fitted_values, residuals = _arima_original_scale_fitted(model, y)
    elapsed = time.perf_counter() - start

    final_model = model.final_model_
    params = {
        "p": getattr(model, "p_", math.nan),
        "d": getattr(model, "d_", math.nan),
        "q": getattr(model, "q_", math.nan),
        "use_constant": getattr(model, "constant_term_", None),
        "c": getattr(final_model, "c_", math.nan),
        "phi": getattr(final_model, "phi_", []),
        "theta": getattr(final_model, "theta_", []),
    }
    fit_stats = {
        "fit_aic": getattr(model, "fit_aic_", math.nan),
        "aic": getattr(final_model, "aic_", math.nan),
        "max_p": getattr(model, "max_p", math.nan),
        "max_d": getattr(model, "max_d", math.nan),
        "max_q": getattr(model, "max_q", math.nan),
        "internal_fitted_length": len(getattr(final_model, "fitted_values_", [])),
        "internal_residual_length": len(getattr(final_model, "residuals_", [])),
    }
    model_name = (
        f"ARIMA({int(model.p_)},{int(model.d_)},{int(model.q_)})"
        f"{'+c' if model.constant_term_ else ''}"
    )
    return FitResult(
        forecast=forecast,
        fitted_values=fitted_values,
        residuals=residuals,
        train_seconds=train_seconds,
        elapsed_seconds=elapsed,
        status="ok",
        model=model_name,
        params=params,
        fit_stats=fit_stats,
    )


def _dotm_forecast_from_fit(model, h: int) -> np.ndarray:
    """Forecast DOTM from the fitted state without refitting."""
    from aeon.forecasting.stats._dotm import (
        _dotm_forecast_from_state,
        _seasonal_forecast,
    )

    n = model._y_adjusted_.shape[0]
    adjusted = _dotm_forecast_from_state(
        n,
        int(h),
        model.level_,
        model.a_,
        model.b_,
        model.mean_y_,
        model.alpha_,
        model.theta_,
    )
    if not model.deseasonalised_:
        return adjusted
    seasonal = _seasonal_forecast(model.seasonal_factors_, h, model.season_length_, n)
    if model.decomposition_type_ == "additive":
        return adjusted + seasonal
    return adjusted * seasonal


def _fit_dotm(y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit aeon DOTM once and return reusable outputs."""
    from aeon.forecasting.stats import DOTM

    start = time.perf_counter()
    model = DOTM(season_length=max(1, int(period)), seasonal_test="auto").fit(y)
    train_seconds = time.perf_counter() - start
    forecast = _dotm_forecast_from_fit(model, h)
    elapsed = time.perf_counter() - start
    params = {
        "initial_level": getattr(model, "initial_level_", math.nan),
        "alpha": getattr(model, "alpha_", math.nan),
        "theta": getattr(model, "theta_", math.nan),
        "season_length": getattr(model, "season_length_", math.nan),
        "decomposition_type": getattr(model, "decomposition_type_", None),
        "deseasonalised": getattr(model, "deseasonalised_", None),
    }
    fit_stats = {
        "sse": getattr(model, "sse_", math.nan),
        "original_scale_sse": getattr(model, "original_scale_sse_", math.nan),
    }
    model_name = "seasonal" if getattr(model, "deseasonalised_", False) else "N"
    return FitResult(
        forecast=forecast,
        fitted_values=_as_float_array(model.fitted_values_),
        residuals=_as_float_array(model.residuals_),
        train_seconds=train_seconds,
        elapsed_seconds=elapsed,
        status="ok",
        model=model_name,
        params=params,
        fit_stats=fit_stats,
    )


def _sf_model_dict_stats(model_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract compact scalar stats from a statsforecast model dictionary."""
    stat_keys = [
        "loglik",
        "aic",
        "bic",
        "aicc",
        "mse",
        "amse",
        "sigma2",
        "n_params",
        "m",
        "n",
        "nobs",
        "n_cond",
        "method",
        "modeltype",
        "seasontype",
        "code",
    ]
    return {key: model_dict[key] for key in stat_keys if key in model_dict}


def _sf_model_params(model, model_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract compact selected-model parameters from statsforecast."""
    param_keys = [
        "par",
        "coef",
        "components",
        "arma",
        "model",
        "mean_y",
        "seasontype",
        "modeltype",
    ]
    params = {
        "class": type(model).__name__,
        "alias": getattr(model, "alias", None),
        "season_length": getattr(model, "season_length", None),
    }
    for key in param_keys:
        if key in model_dict:
            params[key] = model_dict[key]
    return params


def _sf_fitted_and_residuals(
    model_dict: dict[str, Any], y: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return statsforecast fitted values and residuals where available."""
    fitted = model_dict.get("fitted")
    residuals = model_dict.get("actual_residuals", model_dict.get("residuals"))

    fitted_arr = None if fitted is None else _as_float_array(fitted)
    residual_arr = None if residuals is None else _as_float_array(residuals)

    if fitted_arr is None and residual_arr is not None:
        x = model_dict.get("x")
        actual = y if x is None else _as_float_array(x)
        if residual_arr.size <= actual.size:
            fitted_arr = actual[-residual_arr.size :] - residual_arr

    if residual_arr is None and fitted_arr is not None and fitted_arr.size <= y.size:
        residual_arr = y[-fitted_arr.size :] - fitted_arr

    return fitted_arr, residual_arr


def _fit_statsforecast_model(model, y: np.ndarray, h: int) -> FitResult:
    """Fit a statsforecast model once and return reusable outputs."""
    start = time.perf_counter()
    model.fit(y)
    train_seconds = time.perf_counter() - start
    forecast = _as_float_array(model.predict(int(h))["mean"])
    elapsed = time.perf_counter() - start

    model_dict = getattr(model, "model_", {})
    fitted_values, residuals = _sf_fitted_and_residuals(model_dict, y)
    params = _sf_model_params(model, model_dict)
    fit_stats = _sf_model_dict_stats(model_dict)
    model_name = str(
        model_dict.get(
            "method",
            model_dict.get(
                "modeltype",
                model_dict.get(
                    "seasontype", getattr(model, "model", type(model).__name__)
                ),
            ),
        )
    )
    return FitResult(
        forecast=forecast,
        fitted_values=fitted_values,
        residuals=residuals,
        train_seconds=train_seconds,
        elapsed_seconds=elapsed,
        status="ok",
        model=model_name,
        params=params,
        fit_stats=fit_stats,
    )


def _fit_statsforecast(name: str, y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit a named statsforecast equivalent and return persistent outputs."""
    from statsforecast.models import (
        AutoARIMA,
        AutoCES,
        AutoETS,
        DynamicOptimizedTheta,
    )

    season_length = max(1, int(period))
    if name in {"ARIMA", "AutoARIMA"}:
        model = AutoARIMA(season_length=season_length)
    elif name in {"ETS", "AutoETS"}:
        model = AutoETS(season_length=season_length)
    elif name in {"CES", "AutoCES"}:
        model = AutoCES(season_length=season_length)
    elif name == "DOTM":
        model = DynamicOptimizedTheta(season_length=season_length)
    else:
        raise ValueError(f"Unknown statsforecast forecaster {name!r}.")
    return _fit_statsforecast_model(model, y, h)


def _resolve_forecaster_window(y: np.ndarray, h: int, period: int) -> int:
    """Choose a conservative default lookback for window-based forecasters."""
    return max(1, min(y.size - 1, max(int(h) * 2, int(period) * 2, 8)))


def _resolve_regression_window(y: np.ndarray, h: int) -> int:
    """Choose the regression window from the series length."""
    series_length = int(y.size)
    window = max(12, min(500, series_length // 4))
    max_valid_window = max(1, series_length - int(h) - 2)
    return min(window, max_valid_window)


def _aeon_forecast_from_fitted(model, y: np.ndarray, h: int) -> np.ndarray:
    """Forecast ``h`` steps from one fitted aeon model."""
    if hasattr(model, "series_to_series_forecast"):
        return _as_float_array(model.series_to_series_forecast(y, prediction_horizon=h))
    if hasattr(model, "iterative_forecast"):
        return _as_float_array(model.iterative_forecast(y, prediction_horizon=h))
    if hasattr(model, "direct_forecast"):
        return _as_float_array(model.direct_forecast(y, prediction_horizon=h))

    preds = np.empty(int(h), dtype=np.float64)
    history = _as_float_array(y)
    for i in range(int(h)):
        preds[i] = float(model.predict(history))
        history = np.append(history, preds[i])
    return preds


def _coerce_fitted_arrays(
    y: np.ndarray, fitted_values, residuals
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert fitted/residual attributes to arrays and derive one from the other."""
    fitted_arr = None if fitted_values is None else _as_float_array(fitted_values)
    residual_arr = None if residuals is None else _as_float_array(residuals)

    if fitted_arr is None and residual_arr is not None and residual_arr.size <= y.size:
        fitted_arr = y[-residual_arr.size :] - residual_arr
    if residual_arr is None and fitted_arr is not None and fitted_arr.size <= y.size:
        residual_arr = y[-fitted_arr.size :] - fitted_arr
    return fitted_arr, residual_arr


def _build_aeon_forecaster(name: str, y: np.ndarray, h: int, period: int):
    """Construct an aeon forecaster with sensible defaults for this experiment."""
    window = _resolve_forecaster_window(y, h, period)
    regression_window = _resolve_regression_window(y, h)
    season_length = max(1, int(period))

    if name == "NaiveForecaster":
        from aeon.forecasting import NaiveForecaster

        return NaiveForecaster(seasonal_period=season_length)
    if name == "RegressionForecaster":
        from aeon.forecasting import RegressionForecaster

        return RegressionForecaster(window=regression_window, regressor=Ridge())
    if name == "DrCIF":
        from aeon.forecasting import RegressionForecaster
        from aeon.regression.interval_based import DrCIFRegressor

        return RegressionForecaster(
            window=regression_window,
            regressor=DrCIFRegressor(n_estimators=5, random_state=0, n_jobs=1),
        )
    if name == "ARIMA":
        from aeon.forecasting.stats import ARIMA

        return ARIMA()
    if name == "AutoARIMA":
        from aeon.forecasting.stats import AutoARIMA

        return AutoARIMA()
    if name == "ETS":
        from aeon.forecasting.stats import ETS

        return ETS(seasonal_period=season_length)
    if name == "AutoETS":
        from aeon.forecasting.stats import AutoETS

        return AutoETS(seasonal_period=season_length)
    if name == "CES":
        from aeon.forecasting.stats import CES

        return CES(season_length=season_length)
    if name == "AutoCES":
        from aeon.forecasting.stats import AutoCES

        return AutoCES(season_length=season_length)
    if name == "DOTM":
        from aeon.forecasting.stats import DOTM

        return DOTM(season_length=season_length)
    if name == "TAR":
        from aeon.forecasting.stats import TAR

        return TAR()
    if name == "AutoTAR":
        from aeon.forecasting.stats import AutoTAR

        return AutoTAR()
    if name == "Theta":
        from aeon.forecasting.stats import Theta

        return Theta()
    if name == "TVP":
        from aeon.forecasting.stats import TVP

        return TVP(window=regression_window)
    if name == "SETAR":
        from aeon.forecasting.machine_learning import SETAR

        return SETAR()
    if name == "SETARTree":
        from aeon.forecasting.machine_learning import SETARTree

        return SETARTree()
    if name == "SETARForest":
        from aeon.forecasting.machine_learning import SETARForest

        return SETARForest()
    if name == "TCNForecaster":
        from aeon.forecasting.deep_learning import TCNForecaster

        return TCNForecaster(window=window, horizon=h)
    if name == "DeepARForecaster":
        from aeon.forecasting.deep_learning import DeepARForecaster

        return DeepARForecaster(window=window, horizon=h)
    if name == "NBeatsForecaster":
        from aeon.forecasting.deep_learning import NBeatsForecaster

        return NBeatsForecaster(window=regression_window, horizon=h)
    raise ValueError(f"Unknown aeon forecaster {name!r}.")


def _fit_generic_aeon(name: str, y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit an aeon forecaster through the public forecasting API."""
    start = time.perf_counter()
    model = _build_aeon_forecaster(name, y, h, period)
    model.fit(y)
    train_seconds = time.perf_counter() - start
    forecast = _aeon_forecast_from_fitted(model, y, int(h))
    elapsed = time.perf_counter() - start
    fitted_values, residuals = _coerce_fitted_arrays(
        y,
        getattr(model, "fitted_values_", None),
        getattr(model, "residuals_", None),
    )
    fit_stats = {}
    if hasattr(model, "fit_time_millis_"):
        fit_stats["fit_time_millis"] = model.fit_time_millis_
    if hasattr(model, "aic_"):
        fit_stats["aic"] = model.aic_
    if hasattr(model, "fit_aic_"):
        fit_stats["fit_aic"] = model.fit_aic_
    if hasattr(model, "best_model_name_"):
        fit_stats["best_model_name"] = model.best_model_name_
    return FitResult(
        forecast=forecast,
        fitted_values=fitted_values,
        residuals=residuals,
        train_seconds=train_seconds,
        elapsed_seconds=elapsed,
        status="ok",
        model=type(model).__name__,
        params=model.get_params(deep=False) if hasattr(model, "get_params") else {},
        fit_stats=fit_stats,
    )


def _fit_aeon(name: str, y: np.ndarray, h: int, period: int) -> FitResult:
    """Fit a named aeon forecaster and return persistent outputs."""
    canonical = FORECASTER_ALIASES[name]
    if canonical == "AutoARIMA":
        return _fit_arima(y, h, period)
    if canonical == "AutoETS":
        return _fit_ets(y, h, period)
    if canonical == "AutoCES":
        return _fit_ces(y, h, period)
    if canonical == "DOTM":
        return _fit_dotm(y, h, period)
    return _fit_generic_aeon(canonical, y, h, period)


def fit_forecaster(
    backend: str, name: str, y: np.ndarray, h: int, period: int
) -> FitResult:
    """Fit a named backend/forecaster pair and return persistent outputs."""
    canonical_backend = BACKEND_ALIASES[backend]
    canonical_name = FORECASTER_ALIASES[name]
    try:
        if canonical_backend == "aeon":
            return _fit_aeon(canonical_name, y, h, period)
        if canonical_backend == "statsforecast":
            return _fit_statsforecast(canonical_name, y, h, period)
    except Exception as exc:  # noqa: BLE001
        return FitResult(
            forecast=None,
            fitted_values=None,
            residuals=None,
            train_seconds=math.nan,
            elapsed_seconds=math.nan,
            status="error",
            model="",
            params={},
            fit_stats={},
            error=repr(exc),
        )
    raise ValueError(f"Unknown backend {backend!r}.")


def _existing_series_ids(path: Path) -> set[str]:
    """Read existing series IDs from a result file after ``@results``."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    in_results = False
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            if not in_results:
                if line.strip() == "@results":
                    in_results = True
                    header = next(fh, "")
                    reader = csv.DictReader(fh, fieldnames=next(csv.reader([header])))
                    for row in reader:
                        if row.get("SeriesID"):
                            ids.add(row["SeriesID"])
                    break
    return ids


def _write_header(
    path: Path,
    *,
    competition: str,
    backend: str,
    forecaster: str,
    split: str,
    columns: list[str],
    args: argparse.Namespace,
):
    """Write the metadata header and CSV column line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "# persistent forecasting result file",
        f"# @competition {competition}",
        f"# @backend {backend}",
        f"# @forecaster {forecaster}",
        f"# @split {split}",
        f"# @generated_utc {datetime.now(timezone.utc).isoformat()}",
        "# @format CSV after @results",
        "# @parameter SeriesID: '<competition>:<index>' stable fcompdata key",
        "# @parameter frequency: fcompdata series type, e.g. yearly/monthly/hourly",
        "# @parameter seasonality: integer seasonal period supplied to the forecaster",
        "# @parameter train_seconds: model fit time only",
        "# @parameter fit_seconds: total per-series adapter time including output "
        "generation",
        "# @parameter model_params_json: selected model and fitted parameter values",
        "# @parameter fit_stats_json: training objective/IC/statistics for later "
        "ensembles",
        f"# @script_args {_json_dumps(vars(args))}",
        f"# @columns {','.join(columns)}",
        "@results",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(header_lines))
        fh.write("\n")
        writer = csv.writer(fh)
        writer.writerow(columns)


def _append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]):
    """Append result rows using ``columns`` order."""
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        for row in rows:
            writer.writerow(row)


def _prepare_output_file(
    path: Path,
    *,
    competition: str,
    backend: str,
    forecaster: str,
    split: str,
    columns: list[str],
    args: argparse.Namespace,
) -> set[str]:
    """Create or resume an output file and return existing completed IDs."""
    if args.overwrite and path.exists():
        path.unlink()
    if path.exists() and args.resume:
        return _existing_series_ids(path)
    _write_header(
        path,
        competition=competition,
        backend=backend,
        forecaster=forecaster,
        split=split,
        columns=columns,
        args=args,
    )
    return set()


def _has_full_results(paths: tuple[Path, Path], expected_ids: set[str]) -> bool:
    """Return whether all expected series are present in both output files."""
    if not expected_ids:
        return False
    return all(expected_ids <= _existing_series_ids(path) for path in paths)


def build_rows(
    item, backend: str, result: FitResult, max_h: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one test row and one train row."""
    y_train = _as_float_array(item.y_train)
    y_test = _as_float_array(item.y_test)
    forecast = result.forecast
    test_metrics = _metrics(y_train, y_test, forecast, item.period)
    train_metrics = _train_metrics(y_train, result.fitted_values, item.period)
    fitted_length = 0
    fitted_start = ""
    if result.fitted_values is not None:
        fitted_length = int(_as_float_array(result.fitted_values).size)
        if fitted_length > 0 and fitted_length <= y_train.size:
            fitted_start = int(y_train.size - fitted_length)

    base = {
        "SeriesID": item.key,
        "backend": backend,
        "frequency": item.series_type,
        "seasonality": item.period,
        "status": result.status,
        "train_seconds": _blankable(result.train_seconds),
        "fit_seconds": _blankable(result.elapsed_seconds),
        "model": result.model,
        "model_params_json": _json_dumps(result.params),
        "fit_stats_json": _json_dumps(result.fit_stats),
        "error": result.error,
    }
    test_row = {
        **base,
        "horizon": item.h,
        "smape": _blankable(test_metrics["smape"]),
        "mase": _blankable(test_metrics["mase"]),
        "mae": _blankable(test_metrics["mae"]),
        "rmse": _blankable(test_metrics["rmse"]),
    }
    for i in range(max_h):
        test_row[f"pred_{i + 1}"] = (
            _blankable(forecast[i])
            if forecast is not None and i < forecast.shape[0]
            else ""
        )
        test_row[f"actual_{i + 1}"] = (
            _blankable(y_test[i]) if i < y_test.shape[0] else ""
        )

    train_row = {
        **base,
        "n_train": y_train.shape[0],
        "fitted_start": fitted_start,
        "fitted_length": fitted_length,
        "fitted_values": _encode_vector(result.fitted_values),
        "residuals": _encode_vector(result.residuals),
        "train_smape": _blankable(train_metrics["smape"]),
        "train_mase": _blankable(train_metrics["mase"]),
        "train_mae": _blankable(train_metrics["mae"]),
        "train_rmse": _blankable(train_metrics["rmse"]),
    }
    return test_row, train_row


def _fit_item_rows(item, backend: str, forecaster: str, max_h: int):
    """Fit one backend/forecaster/series job and build result rows."""
    result = fit_forecaster(
        backend,
        forecaster,
        _as_float_array(item.y_train),
        int(item.h),
        int(item.period),
    )
    test_row, train_row = build_rows(item, backend, result, max_h)
    return item.key, test_row, train_row


def _dataset_sort_key(name: str) -> tuple[int, str]:
    """Sort datasets with M4 last."""
    return (1 if name == "M4" else 0, name)


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["M1", "M3"],
        help="M-competition datasets to include.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=[],
        help="Optional series types to include, for example yearly monthly.",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        default=10,
        help=(
            "Number of series per dataset/type before explicit inclusions. "
            "Use 0 for all."
        ),
    )
    parser.add_argument(
        "--include-indices",
        nargs="*",
        default=[],
        help="Explicit DATASET:INDEX cases to include after the stratified sample.",
    )
    parser.add_argument(
        "--forecasters",
        nargs="+",
        default=list(DEFAULT_FORECASTERS),
        choices=sorted(FORECASTER_ALIASES),
        help=(
            "Forecasters to run. Aeon accepts the current exported forecasters; "
            "statsforecast only supports ARIMA/AutoARIMA, ETS/AutoETS, "
            "CES/AutoCES, and DOTM."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=sorted(BACKEND_ALIASES),
        help="Backend implementations to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for persistent result files.",
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        default=DEFAULT_DATA_HOME,
        help=(
            "Local fcompdata cache directory. Default: AEON_FCOMPDATA_HOME or "
            "experiments/data/fcompdata."
        ),
    )
    parser.add_argument(
        "--m4-home",
        type=Path,
        default=DEFAULT_M4_HOME,
        help=(
            "Directory containing M4 TSF files. Default: AEON_M4_HOME or "
            "D:\\Data\\Forecasting\\m4."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace incomplete result files instead of appending/resuming. "
            "Complete Test/Train result pairs are preserved."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip series IDs already present in existing output files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Progress print interval.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Total worker threads to divide across concurrent backend/forecaster "
            "runs for a dataset. Each run receives an equal share of threads and "
            f"processes series in batches of {SERIES_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Write only the test results file and skip the train results file.",
    )
    return parser.parse_args()


def _append_completed_row(
    *,
    series_id: str,
    test_row: dict[str, Any],
    train_row: dict[str, Any],
    test_done: set[str],
    train_done: set[str],
    test_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    write_train: bool,
) -> None:
    """Queue completed test/train rows if that split is missing the series."""
    if series_id not in test_done:
        test_rows.append(test_row)
    if write_train and series_id not in train_done:
        train_rows.append(train_row)


def _run_pending_jobs(
    *,
    pending: list[SelectedSeries],
    dataset: str,
    backend: str,
    forecaster: str,
    max_h: int,
    test_done: set[str],
    test_path: Path,
    test_columns: list[str],
    write_train: bool,
    train_done: set[str] | None,
    train_path: Path | None,
    train_columns: list[str] | None,
    progress_every: int,
    quiet: bool,
    n_jobs: int,
) -> None:
    """Run per-series jobs in a bounded thread pool and append CSV rows."""
    test_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    def flush_if_needed(force: bool = False) -> None:
        if test_rows and (force or len(test_rows) >= 25):
            _append_rows(test_path, test_columns, test_rows)
            test_rows.clear()
        if write_train and train_path is not None and train_columns is not None:
            if train_rows and (force or len(train_rows) >= 25):
                _append_rows(train_path, train_columns, train_rows)
                train_rows.clear()

    def record_completion(series_id: str, test_row, train_row, completed: int) -> None:
        _append_completed_row(
            series_id=series_id,
            test_row=test_row,
            train_row=train_row,
            test_done=test_done,
            train_done=train_done if train_done is not None else set(),
            test_rows=test_rows,
            train_rows=train_rows,
            write_train=write_train,
        )
        flush_if_needed()
        if not quiet and (
            completed % max(1, progress_every) == 0 or completed == len(pending)
        ):
            print(  # noqa: T201
                f"{dataset} {backend} {forecaster}: completed "
                f"{completed}/{len(pending)} new series"
            )

    if n_jobs == 1:
        for completed, item in enumerate(pending, start=1):
            series_id, test_row, train_row = _fit_item_rows(
                item, backend, forecaster, max_h
            )
            record_completion(series_id, test_row, train_row, completed)
        flush_if_needed(force=True)
        return

    chunks = [
        pending[i : i + SERIES_BATCH_SIZE]
        for i in range(0, len(pending), SERIES_BATCH_SIZE)
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for chunk_results in executor.map(
            lambda chunk: [
                _fit_item_rows(item, backend, forecaster, max_h) for item in chunk
            ],
            chunks,
        ):
            for series_id, test_row, train_row in chunk_results:
                completed += 1
                record_completion(series_id, test_row, train_row, completed)
    flush_if_needed(force=True)


def _run_estimator_job(
    *,
    dataset: str,
    items: list[SelectedSeries],
    backend: str,
    forecaster: str,
    max_h: int,
    test_columns: list[str],
    train_columns: list[str],
    args: argparse.Namespace,
    n_jobs: int,
) -> tuple[Path, Path | None] | None:
    """Run one backend/forecaster output set for a dataset."""
    forecaster_dir = args.output_dir / backend / forecaster
    test_path = forecaster_dir / f"{dataset}_{forecaster}_Test.csv"
    train_path = forecaster_dir / f"{dataset}_{forecaster}_Train.csv"
    expected_ids = {item.key for item in items}
    result_paths = (test_path,) if args.no_train else (test_path, train_path)
    if _has_full_results(result_paths, expected_ids):
        if not args.quiet:
            print(  # noqa: T201
                f"Skipping {dataset} {backend} {forecaster}: existing full "
                f"{'Test' if args.no_train else 'Test/Train'} results present"
            )
        return None

    test_done = _prepare_output_file(
        test_path,
        competition=dataset,
        backend=backend,
        forecaster=forecaster,
        split="test",
        columns=test_columns,
        args=args,
    )
    train_done = None
    if not args.no_train:
        train_done = _prepare_output_file(
            train_path,
            competition=dataset,
            backend=backend,
            forecaster=forecaster,
            split="train",
            columns=train_columns,
            args=args,
        )

    pending = [
        item
        for item in items
        if item.key not in test_done
        or (train_done is not None and item.key not in train_done)
    ]
    _run_pending_jobs(
        pending=pending,
        dataset=dataset,
        backend=backend,
        forecaster=forecaster,
        max_h=max_h,
        test_done=test_done,
        test_path=test_path,
        test_columns=test_columns,
        write_train=not args.no_train,
        train_done=train_done,
        train_path=None if args.no_train else train_path,
        train_columns=None if args.no_train else train_columns,
        progress_every=args.progress_every,
        quiet=args.quiet,
        n_jobs=n_jobs,
    )
    return (test_path, None if args.no_train else train_path)


def main() -> int:
    """Run the persistence export."""
    args = parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1.")
    _patch_fcompdata_home(args.data_home)
    selected = select_experiment_series(args)
    by_dataset = defaultdict(list)
    for item in selected:
        by_dataset[item.dataset].append(item)

    for dataset, items in sorted(
        by_dataset.items(), key=lambda pair: _dataset_sort_key(pair[0])
    ):
        max_h = max(int(item.h) for item in items)
        test_columns = (
            [
                "SeriesID",
                "backend",
                "frequency",
                "seasonality",
                "horizon",
                "status",
                "train_seconds",
                "fit_seconds",
            ]
            + [f"pred_{i + 1}" for i in range(max_h)]
            + [f"actual_{i + 1}" for i in range(max_h)]
            + [
                "smape",
                "mase",
                "mae",
                "rmse",
                "model",
                "model_params_json",
                "fit_stats_json",
                "error",
            ]
        )
        train_columns = [
            "SeriesID",
            "backend",
            "frequency",
            "seasonality",
            "n_train",
            "fitted_start",
            "fitted_length",
            "status",
            "train_seconds",
            "fit_seconds",
            "fitted_values",
            "residuals",
            "train_smape",
            "train_mase",
            "train_mae",
            "train_rmse",
            "model",
            "model_params_json",
            "fit_stats_json",
            "error",
        ]

        estimator_jobs = [
            (BACKEND_ALIASES[requested_backend], FORECASTER_ALIASES[requested])
            for requested_backend in args.backends
            for requested in args.forecasters
        ]
        per_estimator_jobs = max(1, args.n_jobs // max(1, len(estimator_jobs)))

        with ThreadPoolExecutor(max_workers=max(1, len(estimator_jobs))) as executor:
            futures = [
                executor.submit(
                    _run_estimator_job,
                    dataset=dataset,
                    items=items,
                    backend=backend,
                    forecaster=forecaster,
                    max_h=max_h,
                    test_columns=test_columns,
                    train_columns=train_columns,
                    args=args,
                    n_jobs=per_estimator_jobs,
                )
                for backend, forecaster in estimator_jobs
            ]

            for future in futures:
                paths = future.result()
                if paths is None or args.quiet:
                    continue
                test_path, train_path = paths
                print(f"Wrote {test_path}")  # noqa: T201
                if train_path is not None:
                    print(f"Wrote {train_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
