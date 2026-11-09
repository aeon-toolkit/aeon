"""Tests for STLSeriesTransformer."""

import numpy as np
import pytest

from aeon.transformations.series._stl import STLSeriesTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies


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
        STLSeriesTransformer(period=12, low_pass_jump=0).fit_transform(y)
    with pytest.raises(ValueError):
        STLSeriesTransformer(period=12, output="foo").fit_transform(y)


@pytest.mark.parametrize("n", [10, 23, 24, 25, 50])
@pytest.mark.parametrize("jump", [2, 3, 5, 7])
def test_interpolate_knots_sets_every_point(n, jump):
    """Every point is interpolated, including the gap before the last knot."""
    from aeon.transformations.series._stl import _interpolate_knots_nb

    line = 2.0 * np.arange(n) + 1.0
    ys = np.full(n, -999.0)
    knots = sorted(set(range(0, n, jump)) | {n - 1})
    ys[knots] = line[knots]
    _interpolate_knots_nb(ys, n, jump)
    assert not np.any(ys == -999.0)
    np.testing.assert_allclose(ys, line)


def test_trend_jump_leaves_no_unset_value():
    """Regression test for #3980: trend_jump=3 left index 22 of 24 unset."""
    y = np.arange(24.0) + 1
    trend = STLSeriesTransformer(
        period=4, seasonal=7, trend=7, trend_jump=3, inner_iter=1, output="trend"
    ).fit_transform(y)
    trend = np.asarray(trend).ravel()
    assert np.all(np.isfinite(trend))
    assert trend[-3] > 0.0
    assert min(trend[-4], trend[-2]) <= trend[-3] <= max(trend[-4], trend[-2])


@pytest.mark.parametrize("n", [24, 25, 26, 50])
@pytest.mark.parametrize("kwargs", [{"trend_jump": 3}, {"seasonal_jump": 3}])
def test_jumps_keep_components_in_range(n, kwargs):
    """A non-default jump must not leave zeros or garbage in the output."""
    y = np.arange(n, dtype=float) + 100.0
    trend = STLSeriesTransformer(
        period=4, seasonal=7, trend=7, inner_iter=1, output="trend", **kwargs
    ).fit_transform(y)
    trend = np.asarray(trend).ravel()
    assert np.all(np.isfinite(trend))
    assert np.all(trend > 99.0) and np.all(trend < 100.0 + n)


def test_straight_line_trend_exact():
    """Regression test for #3994: trend of straight line should match line exactly."""
    params = dict(period=4, seasonal=7, trend=9, low_pass=5)
    y = np.arange(48, dtype=float)

    trend = STLSeriesTransformer(
        **params, inner_iter=1, outer_iter=0, output="trend"
    ).fit_transform(y)
    np.testing.assert_allclose(np.ravel(trend), y, atol=1e-10)


def test_low_pass_parameters_affect_output():
    """Regression test for #3994: changing low_pass parameters must affect output."""
    rng = np.random.default_rng(0)
    t = np.arange(48)
    z = 0.05 * t + np.sin(2 * np.pi * t / 4) + 0.1 * rng.normal(size=48)

    a = STLSeriesTransformer(period=4, low_pass=5, output="seasonal").fit_transform(z)
    b = STLSeriesTransformer(
        period=4, low_pass=21, low_pass_deg=0, output="seasonal"
    ).fit_transform(z)
    assert not np.allclose(np.ravel(a), np.ravel(b))


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if statsmodels is not available",
)
def test_matches_statsmodels_stl():
    """Regression test for #3994: verify close agreement with statsmodels STL."""
    from statsmodels.tsa.seasonal import STL

    params = dict(period=4, seasonal=7, trend=9, low_pass=5)
    rng = np.random.default_rng(42)
    t = np.arange(48)
    z = 0.05 * t + np.sin(2 * np.pi * t / 4) + 0.1 * rng.normal(size=48)

    aeon_res = STLSeriesTransformer(
        **params, inner_iter=2, outer_iter=0, output="all"
    ).fit_transform(z)
    sm_res = STL(z, **params).fit(inner_iter=2, outer_iter=0)

    np.testing.assert_allclose(aeon_res[:, 0], sm_res.seasonal, atol=1e-10)
    np.testing.assert_allclose(aeon_res[:, 1], sm_res.trend, atol=1e-10)
