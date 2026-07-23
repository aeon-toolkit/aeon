"""Tests for MSTLSeriesTransformer."""

import numpy as np
import pytest

from aeon.transformations.series._mstl import MSTLSeriesTransformer
from aeon.transformations.series._stl import STLSeriesTransformer


def _toy_two_season(n=504):
    """Toy signal with daily (24) and weekly (168) seasonality + trend."""
    t = np.arange(n, dtype=float)
    y = 1.5 * np.sin(2 * np.pi * t / 24) + 0.75 * np.sin(2 * np.pi * t / 168) + 0.01 * t
    return y.astype(float)


def test_mstl_all_output_shape_and_components():
    """The all output has the expected shape and stored components."""
    periods = [24, 168]
    y = _toy_two_season()
    mstl = MSTLSeriesTransformer(periods=periods, iterate=1, output="all")
    Z = mstl.fit_transform(y)
    n = len(y)
    n_components = len(periods) + 2
    assert isinstance(Z, np.ndarray) and Z.shape == (n, n_components)

    assert set(mstl.components_.keys()) == {
        "periods",
        "seasonals",
        "seasonal_sum",
        "trend",
        "remainder",
    }
    assert len(mstl.components_["seasonals"]) == len(periods)
    for s in mstl.components_["seasonals"]:
        assert s.shape == (n,)


def test_modes_shapes():
    """Check return shapes across all output modes."""
    periods = [24, 168]
    y = _toy_two_season()
    n = len(y)
    for output in ["remainder", "trend", "seasonal_sum", "seasonals", "all"]:
        mstl = MSTLSeriesTransformer(periods=periods, iterate=1, output=output)
        Z = mstl.fit_transform(y)
        if output in {"remainder", "trend", "seasonal_sum"}:
            assert Z.shape == (n,)
        elif output == "seasonals":
            assert Z.shape == (n, len(periods))
        elif output == "all":
            assert Z.shape == (n, len(periods) + 2)


def test_mstl_matches_stl_single_season():
    """With one period and iterate=1, MSTL reduces to STL (same internals)."""
    n = 480
    period = 24
    t = np.arange(n, dtype=float)
    y = np.sin(2 * np.pi * t / period) + 0.02 * t

    mstl = MSTLSeriesTransformer(
        periods=[period], iterate=1, s_windows=[11], output="all"
    )
    Zm = mstl.fit_transform(y)  # columns: [seasonal, trend, remainder]

    stl = STLSeriesTransformer(period=period, seasonal=11, output="all")
    Zs = stl.fit_transform(y)  # columns: [seasonal, trend, resid]

    assert np.allclose(Zm[:, 0], Zs[:, 0], atol=1e-8)
    assert np.allclose(Zm[:, 1], Zs[:, 1], atol=1e-8)
    assert np.allclose(Zm[:, 2], Zs[:, 2], atol=1e-8)


def test_invalid_periods_raises():
    """All candidate periods are filtered out (>= n/2) -> ValueError."""
    n = 200
    invalid_periods = [150, 120]
    rng = np.random.default_rng(0)
    y = rng.random(n)
    with pytest.raises(ValueError):
        MSTLSeriesTransformer(periods=invalid_periods).fit_transform(y)


def test_iterate_validation():
    """Iterate must be >=1."""
    n = 240
    period = 12
    rng = np.random.default_rng(1)
    y = rng.random(n)
    with pytest.raises(ValueError):
        MSTLSeriesTransformer(periods=[period], iterate=0).fit_transform(y)


def test_mstl_imputes_missing_values_before_decomposition():
    """Built-in imputation matches decomposition of explicitly interpolated data."""
    period = 24
    missing_index = 100
    y = _toy_two_season(n=period * 10)
    y_with_missing = y.copy()
    y_with_missing[missing_index] = np.nan
    y_interpolated = y.copy()
    y_interpolated[missing_index] = (y[missing_index - 1] + y[missing_index + 1]) / 2
    mstl_params = {
        "periods": [period],
        "iterate": 1,
        "s_windows": [11],
        "output": "all",
    }

    imputed_components = MSTLSeriesTransformer(
        **mstl_params, impute_missing=True
    ).fit_transform(y_with_missing)
    expected_components = MSTLSeriesTransformer(
        **mstl_params, impute_missing=False
    ).fit_transform(y_interpolated)

    np.testing.assert_allclose(imputed_components, expected_components)


@pytest.mark.parametrize("boxcox_lambda", [None, 0.5])
def test_mstl_rejects_missing_values_without_imputation(boxcox_lambda):
    """Missing values give a clear error when imputation is disabled."""
    y = _toy_two_season(n=24 * 10)
    y[100] = np.nan

    with pytest.raises(ValueError, match="impute_missing=True"):
        MSTLSeriesTransformer(
            periods=[24],
            s_windows=[11],
            boxcox_lambda=boxcox_lambda,
            impute_missing=False,
        ).fit_transform(y)


def test_mstl_with_jumps_reconstructs_input_by_construction():
    """Knot jumps preserve the output components' reconstruction invariant."""
    periods = [24, 168]
    y = _toy_two_season(n=24 * 21)

    components = MSTLSeriesTransformer(
        periods=periods,
        iterate=2,
        s_windows=[11, 15],
        seasonal_jump=2,
        trend_jump=2,
        output="all",
    ).fit_transform(y)

    np.testing.assert_allclose(components.sum(axis=1), y, atol=1e-8)
