"""Tests for MSTLSeriesTransformer."""

import numpy as np
import pytest

from aeon.transformations.series._mstl import MSTLSeriesTransformer
from aeon.transformations.series._stl import STLSeriesTransformer


def _toy_two_season(n=504):
    """Toy signal with daily (24) and weekly (168) seasonality + trend."""
    t = np.arange(n)
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
    t = np.arange(n)
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
