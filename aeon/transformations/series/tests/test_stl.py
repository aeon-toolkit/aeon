"""Tests for STLSeriesTransformer."""

import numpy as np
import pytest

from aeon.transformations.series._stl import STLSeriesTransformer


def _synthetic_series(n=240, period=12, trend_slope=0.01, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / period)
    trend = trend_slope * t
    y = seasonal + trend + (noise * rng.standard_normal(n))
    return y.astype(float), seasonal, trend


def test_shapes_and_columns():
    """Check that STL returns expected shapes and matching component lengths."""
    y, _, _ = _synthetic_series(n=48, period=12, noise=0.0)
    stl = STLSeriesTransformer(period=12, output="all")
    decomp = stl.fit_transform(y)

    # shape checks
    assert isinstance(decomp, np.ndarray)
    assert decomp.ndim == 2 and decomp.shape == (len(y), 3)

    # components_ existence and length
    for k in ["seasonal", "trend", "resid"]:
        assert k in stl.components_
        assert isinstance(stl.components_[k], np.ndarray)
        assert len(stl.components_[k]) == len(y)

    # column order [seasonal, trend, resid]
    assert np.allclose(decomp[:, 0], stl.components_["seasonal"])
    assert np.allclose(decomp[:, 1], stl.components_["trend"])
    assert np.allclose(decomp[:, 2], stl.components_["resid"])


def test_output_modes():
    """Verify that the 'output' parameter controls return shape."""
    y, _, _ = _synthetic_series(n=60, period=12)
    for out in ["all", "seasonal", "trend", "resid"]:
        stl = STLSeriesTransformer(period=12, output=out)
        z = stl.fit_transform(y)
        assert isinstance(z, np.ndarray)
        if out == "all":
            assert z.ndim == 2 and z.shape == (len(y), 3)
        else:
            assert z.ndim == 1 and z.shape == (len(y),)


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
