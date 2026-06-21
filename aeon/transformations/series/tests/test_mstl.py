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


def test_shapes_and_identity():
    """All output has (n, m+2) and components add back to the original series."""
    y = _toy_two_season()
    mstl = MSTLSeriesTransformer(periods=[24, 168], iterate=1, output="all")
    Z = mstl.fit_transform(y)
    n = len(y)
    assert isinstance(Z, np.ndarray) and Z.shape == (
        n,
        4,
    )  # [S24, S168, trend, remainder]

    Ssum = Z[:, 0] + Z[:, 1]
    T = Z[:, 2]
    R = Z[:, 3]
    # y == sum(seasonals) + trend + remainder
    assert np.allclose(y, Ssum + T + R, atol=1e-8, rtol=0.0)

    assert set(mstl.components_.keys()) == {
        "periods",
        "seasonals",
        "seasonal_sum",
        "trend",
        "remainder",
    }
    assert len(mstl.components_["seasonals"]) == 2
    for s in mstl.components_["seasonals"]:
        assert s.shape == (n,)


def test_modes_shapes():
    """Check return shapes across all output modes."""
    y = _toy_two_season()
    m, n = 2, len(y)
    for out in ["remainder", "trend", "seasonal_sum", "seasonals", "all"]:
        mstl = MSTLSeriesTransformer(periods=[24, 168], iterate=1, output=out)
        Z = mstl.fit_transform(y)
        if out in {"remainder", "trend", "seasonal_sum"}:
            assert Z.shape == (n,)
        elif out == "seasonals":
            assert Z.shape == (n, m)
        elif out == "all":
            assert Z.shape == (n, m + 2)


def test_mstl_matches_stl_single_season():
    """With one period and iterate=1, MSTL reduces to STL (same internals)."""
    n = 480
    t = np.arange(n, dtype=float)
    y = np.sin(2 * np.pi * t / 24) + 0.02 * t

    mstl = MSTLSeriesTransformer(periods=[24], iterate=1, s_windows=[11], output="all")
    Zm = mstl.fit_transform(y)  # columns: [seasonal, trend, remainder]

    stl = STLSeriesTransformer(period=24, seasonal=11, output="all")
    Zs = stl.fit_transform(y)  # columns: [seasonal, trend, resid]

    assert np.allclose(Zm[:, 0], Zs[:, 0], atol=1e-8)
    assert np.allclose(Zm[:, 1], Zs[:, 1], atol=1e-8)
    assert np.allclose(Zm[:, 2], Zs[:, 2], atol=1e-8)


def test_invalid_periods_raises():
    """All candidate periods are filtered out (>= n/2) -> ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(200)
    with pytest.raises(ValueError):
        MSTLSeriesTransformer(periods=[150, 120]).fit_transform(y)


def test_iterate_validation():
    """Iterate must be >=1."""
    rng = np.random.default_rng(1)
    y = rng.random(240)
    with pytest.raises(ValueError):
        MSTLSeriesTransformer(periods=[12], iterate=0).fit_transform(y)


def test_mstl_numba_parity_vs_numpy():
    """MSTL using Numba-STL should match NumPy-STL results closely."""
    pytest.importorskip("numba")  # skip if Numba not installed

    n = 24 * 21
    t = np.arange(n, dtype=float)
    y = (
        1.2 * np.sin(2 * np.pi * t / 24 + 0.3)
        + 0.8 * np.sin(2 * np.pi * t / 168)
        + 0.01 * t
    )

    s_windows = [11, 15]

    mstl_np = MSTLSeriesTransformer(
        periods=[24, 168],
        iterate=2,
        s_windows=s_windows,
        output="all",
        stl_use_numba=False,
    )
    mstl_nb = MSTLSeriesTransformer(
        periods=[24, 168],
        iterate=2,
        s_windows=s_windows,
        output="all",
        stl_use_numba=True,
    )

    Z_np = mstl_np.fit_transform(y)
    Z_nb = mstl_nb.fit_transform(y)

    assert np.allclose(Z_nb, Z_np, atol=1e-6, rtol=1e-7)


def test_transform_after_separate_fit():
    """Test transform works after a separate fit call (fit_is_empty path)."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    mstl = MSTLSeriesTransformer(periods=[12], iterate=1)
    mstl.fit(y)
    Xt = mstl.transform(y)
    assert Xt.shape == y.shape


def test_short_series_raises():
    """Test a series shorter than 3 points raises a ValueError."""
    with pytest.raises(ValueError, match="length >= 3"):
        MSTLSeriesTransformer(periods=[2]).fit_transform(np.array([1.0, 2.0]))


def test_impute_missing_interpolates_nans():
    """Test NaNs are linearly interpolated before decomposition."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    y[5] = np.nan
    mstl = MSTLSeriesTransformer(periods=[12], iterate=1, impute_missing=True)
    _, _, remainder, _ = mstl._compute_components_(y)
    assert not np.isnan(remainder).any()


def test_boxcox_lambda_zero_and_nonzero():
    """Test boxcox_lambda applies a Box-Cox transform before decomposition."""
    rng = np.random.default_rng(0)
    y = np.abs(rng.random(240)) + 1.0

    mstl_half = MSTLSeriesTransformer(periods=[12], iterate=1, boxcox_lambda=0.5)
    Xt_half = mstl_half.fit_transform(y)
    assert Xt_half.shape == y.shape

    mstl_zero = MSTLSeriesTransformer(periods=[12], iterate=1, boxcox_lambda=0.0)
    Xt_zero = mstl_zero.fit_transform(y)
    assert Xt_zero.shape == y.shape


def test_boxcox_requires_positive_data():
    """Test a non-positive series with boxcox_lambda set raises a ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(240) - 10
    mstl = MSTLSeriesTransformer(periods=[12], boxcox_lambda=0.5)
    with pytest.raises(ValueError, match="strictly positive"):
        mstl._compute_components_(y)


def test_coerce_1d_passes_through_already_1d_input():
    """Test an already 1D array skips squeezing and is returned unchanged."""
    x = np.array([1.0, 2.0, 3.0])
    result = MSTLSeriesTransformer._coerce_1d(x)
    np.testing.assert_array_equal(result, x)


def test_coerce_1d_rejects_non_squeezable_input():
    """Test a 2D array that cannot be squeezed to 1D raises a ValueError."""
    with pytest.raises(ValueError, match="must be 1D"):
        MSTLSeriesTransformer._coerce_1d(np.zeros((3, 240)))


def test_iterate_must_be_an_integer():
    """Test a non-integer iterate raises a ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    with pytest.raises(ValueError, match="must be an integer"):
        MSTLSeriesTransformer(periods=[12], iterate=1.5).fit_transform(y)


def test_s_windows_length_mismatch_raises():
    """Test s_windows with the wrong length raises a ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    with pytest.raises(ValueError, match="Length of `s_windows`"):
        MSTLSeriesTransformer(periods=[12, 24], s_windows=[11]).fit_transform(y)


def test_s_windows_none_element_uses_default():
    """Test a None entry in s_windows falls back to the computed default."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    mstl = MSTLSeriesTransformer(periods=[12], s_windows=[None])
    Xt = mstl.fit_transform(y)
    assert Xt.shape == y.shape


def test_s_windows_invalid_element_raises():
    """Test an even or too-small s_windows entry raises a ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    with pytest.raises(ValueError, match="must be an odd integer"):
        MSTLSeriesTransformer(periods=[12], s_windows=[4]).fit_transform(y)


def test_periods_must_be_a_non_empty_sequence():
    """Test a non-iterable periods value raises a ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    with pytest.raises(ValueError, match="non-empty sequence"):
        MSTLSeriesTransformer(periods=12).fit_transform(y)


def test_invalid_period_value_raises():
    """Test a period below 2 raises a ValueError."""
    rng = np.random.default_rng(0)
    y = rng.random(240)
    with pytest.raises(ValueError, match="must be integers >= 2"):
        MSTLSeriesTransformer(periods=[1]).fit_transform(y)


def test_na_interp_linear_edge_cases():
    """Test the NaN-interpolation helper's edge cases directly."""
    np.testing.assert_array_equal(
        MSTLSeriesTransformer._na_interp_linear(np.array([])), np.array([])
    )
    no_nan = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        MSTLSeriesTransformer._na_interp_linear(no_nan), no_nan
    )
    with pytest.raises(ValueError, match="all values are NaN"):
        MSTLSeriesTransformer._na_interp_linear(np.array([np.nan, np.nan]))
    interpolated = MSTLSeriesTransformer._na_interp_linear(np.array([1.0, np.nan, 3.0]))
    np.testing.assert_allclose(interpolated, [1.0, 2.0, 3.0])


def test_format_output_with_zero_seasonals():
    """Test _format_output handles an empty seasonals list for every mode."""
    trend = np.zeros(5)
    remainder = np.zeros(5)

    sum_out = MSTLSeriesTransformer(periods=[12], output="seasonal_sum")._format_output(
        [], trend, remainder
    )
    np.testing.assert_array_equal(sum_out, np.zeros(5))

    seasonals_out = MSTLSeriesTransformer(
        periods=[12], output="seasonals"
    )._format_output([], trend, remainder)
    assert seasonals_out.shape == (5, 0)

    all_out = MSTLSeriesTransformer(periods=[12], output="all")._format_output(
        [], trend, remainder
    )
    assert all_out.shape == (5, 2)


def test_format_output_invalid_mode_raises():
    """Test an unrecognised output mode raises a ValueError."""
    transformer = MSTLSeriesTransformer(periods=[12], output="bogus")
    with pytest.raises(ValueError, match="`output` must be one of"):
        transformer._format_output([np.zeros(5)], np.zeros(5), np.zeros(5))


def test_mstl_get_test_params():
    """Test the default test parameters are valid and usable."""
    params = MSTLSeriesTransformer._get_test_params()
    transformer = MSTLSeriesTransformer(**params)
    assert isinstance(transformer, MSTLSeriesTransformer)


def test_mstl_numba_smoke_with_jumps():
    """Numba-STL inside MSTL with knot jumps runs and returns proper shapes."""
    pytest.importorskip("numba")

    y = _toy_two_season(n=24 * 21)

    mstl_nb = MSTLSeriesTransformer(
        periods=[24, 168],
        iterate=2,
        s_windows=[11, 15],
        seasonal_jump=2,
        trend_jump=2,
        output="all",
        stl_use_numba=True,
    )
    Z = mstl_nb.fit_transform(y)
    assert isinstance(Z, np.ndarray) and Z.shape == (len(y), 4)
