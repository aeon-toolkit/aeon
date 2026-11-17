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


def test_robust_with_outlier_block_returns_finite_output():
    """Regression test for #3980: robust=True with a large outlier run was all NaN."""
    t = np.arange(300)
    y = 0.02 * t + 3 * np.sin(2 * np.pi * t / 12)
    y += np.random.default_rng(0).normal(size=300)
    y[100:140] += 1e6
    out = STLSeriesTransformer(
        period=12, seasonal=11, robust=True, output="all"
    ).fit_transform(y)
    assert np.all(np.isfinite(out))


def test_low_pass_parameters_change_output():
    """Regression test for #3980: low_pass settings must affect the result."""
    y, _, _ = _synthetic_series(n=240, period=12, noise=0.5, seed=1)
    a = STLSeriesTransformer(period=12, seasonal=11, output="all").fit_transform(y)
    b = STLSeriesTransformer(
        period=12,
        seasonal=11,
        output="all",
        low_pass=25,
        low_pass_deg=0,
        low_pass_jump=5,
    ).fit_transform(y)
    assert np.abs(a - b).max() > 1e-6


def test_matches_reference_stl_values():
    """Compare with reference values from statsmodels STL (inner_iter=5, no jumps).

    Guards the low-pass LOESS step and the extrapolation of the cycle-subseries.
    """
    y = np.array(
        [1.0, 2.9858, 2.3468, 1.0865, 0.0742, -2.6715, -2.4086, 1.6999, 2.4342,
         3.9444, 2.7525, -1.0307, -0.737, -2.0231, 2.2701, 3.1188, 5.114, 2.5199,
         0.9143, -1.4205, 0.0754, 1.6119, 5.136, 4.5086, 4.5476, 0.2461, 0.6751,
         -0.596, 3.5726, 4.7898, 5.9513, 4.8423, 1.0711, 1.3742, 0.1979, 3.898,
         6.1854, 5.8258, 6.0992, 1.9121]
    )  # fmt: skip
    res = STLSeriesTransformer(
        period=7,
        seasonal=9,
        trend=15,
        low_pass=9,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        inner_iter=5,
        outer_iter=0,
        output="all",
    ).fit_transform(y)
    seasonal, trend = res[:, 0], res[:, 1]
    np.testing.assert_allclose(
        seasonal[:5],
        [1.03276165, 2.26593158, 2.59926639, 0.9494301, -0.82069306],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        trend[:5],
        [0.07402938, 0.16009707, 0.2476795, 0.33578689, 0.4241838],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        seasonal[-3:], [2.27749483, 2.16743398, -2.2460573], atol=1e-6
    )
