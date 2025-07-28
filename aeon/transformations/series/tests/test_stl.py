"""Tests for STLSeriesTransformer."""

import numpy as np
import pandas as pd
import pytest

from aeon.transformations.series._stl import STLSeriesTransformer


def _synthetic_series(n=240, period=12, trend_slope=0.01, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / period)
    trend = trend_slope * t
    y = seasonal + trend + (noise * rng.standard_normal(n))
    idx = pd.date_range("2000-01-01", periods=n, freq="M")
    return pd.Series(y, index=idx), seasonal, trend


def test_shapes_and_columns():
    """Check that STL returns the expected columns and matching component lengths."""
    y, _, _ = _synthetic_series(n=48, period=12, noise=0.0)
    stl = STLSeriesTransformer(period=12, output="all")
    decomp = stl.fit_transform(y)
    assert list(decomp.columns) == ["seasonal", "trend", "resid"]
    assert len(decomp) == len(y)

    for k in ["seasonal", "trend", "resid"]:
        assert k in stl.components_
        assert len(stl.components_[k]) == len(y)


def test_output_modes():
    """Verify that the 'output' parameter controls return type and component naming."""
    y, _, _ = _synthetic_series(n=60, period=12)
    for out in ["all", "seasonal", "trend", "resid"]:
        stl = STLSeriesTransformer(period=12, output=out)
        z = stl.fit_transform(y)
        if out == "all":
            assert isinstance(z, pd.DataFrame) and set(z.columns) == {
                "seasonal",
                "trend",
                "resid",
            }
        else:
            assert isinstance(z, pd.Series)
            assert z.name == out


def test_parameter_validation():
    """Ensure invalid configuration values raise ValueError."""
    y, _, _ = _synthetic_series(n=24, period=12)
    with pytest.raises(ValueError):
        STLSeriesTransformer(period=1).fit_transform(y)
    with pytest.raises(ValueError):
        STLSeriesTransformer(period=12, seasonal=4).fit_transform(y)  # not odd
    with pytest.raises(ValueError):
        STLSeriesTransformer(period=12, trend=11).fit_transform(y)  # trend <= period
    with pytest.raises(ValueError):
        STLSeriesTransformer(period=12, low_pass=11).fit_transform(
            y
        )  # low_pass <= period
    with pytest.raises(ValueError):
        STLSeriesTransformer(period=12, output="foo").fit_transform(y)
