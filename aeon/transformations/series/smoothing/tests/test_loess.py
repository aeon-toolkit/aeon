"""Tests for LoessSmoother."""

import numpy as np

from aeon.transformations.series.smoothing._loess import LoessSmoother

DATA_LINEAR = np.arange(10, dtype=float)

DATA_QUAD = np.arange(10, dtype=float) ** 2

DATA_MULTI = np.array([np.arange(10, dtype=float), np.ones(10, dtype=float) * 5])


def test_loess_linear_recovery():
    """Test that LOESS (degree=1) recovers a straight line perfectly."""
    loess = LoessSmoother(span=0.4, degree=1)
    xt = loess.fit_transform(DATA_LINEAR)

    assert xt.shape == (1, DATA_LINEAR.shape[0])

    np.testing.assert_array_almost_equal(xt[0], DATA_LINEAR, decimal=5)


def test_loess_quadratic_recovery():
    """Test that LOESS (degree=2) recovers a quadratic curve perfectly."""
    loess = LoessSmoother(span=0.5, degree=2)
    xt = loess.fit_transform(DATA_QUAD)

    np.testing.assert_array_almost_equal(xt[0], DATA_QUAD, decimal=5)


def test_multivariate_execution():
    """Test that multivariate inputs are handled correctly per channel."""
    loess = LoessSmoother(span=0.5, degree=1)
    xt = loess.fit_transform(DATA_MULTI)

    assert xt.shape == DATA_MULTI.shape
    np.testing.assert_array_almost_equal(xt[0], DATA_MULTI[0], decimal=5)
    np.testing.assert_array_almost_equal(xt[1], DATA_MULTI[1], decimal=5)


def test_span_extremes():
    """Test execution with extreme span values."""
    loess_full = LoessSmoother(span=1.0, degree=1)
    xt_full = loess_full.fit_transform(DATA_LINEAR)
    np.testing.assert_array_almost_equal(xt_full[0], DATA_LINEAR)

    loess_small = LoessSmoother(span=0.01, degree=1)
    xt_small = loess_small.fit_transform(DATA_LINEAR)
    np.testing.assert_array_almost_equal(xt_small[0], DATA_LINEAR)


def test_smoothing_effectiveness():
    """Test that LOESS actually reduces variance (smooths noise)."""
    x = np.linspace(0, 4 * np.pi, 100)
    y_clean = np.sin(x)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, size=x.shape)
    y_noisy = y_clean + noise

    smoother = LoessSmoother(span=0.2, degree=1)
    y_smoothed = smoother.fit_transform(y_noisy)[0]

    mse_noisy = np.mean((y_noisy - y_clean) ** 2)
    mse_smoothed = np.mean((y_smoothed - y_clean) ** 2)

    assert mse_smoothed < mse_noisy, "LOESS failed to reduce noise variance."


def test_constant_preservation():
    """Test that a constant line remains constant."""
    X = np.ones(50) * 10
    smoother = LoessSmoother(span=0.5, degree=1)
    Xt = smoother.fit_transform(X)[0]

    np.testing.assert_array_almost_equal(Xt, X)


def test_short_series_fallback():
    """Test that very short series are returned as-is without crashing."""
    X = np.array([1.0, 2.0])
    smoother = LoessSmoother(span=0.5, degree=2)
    Xt = smoother.fit_transform(X)[0]

    np.testing.assert_array_equal(Xt, X)


def test_outlier_insensitivity():
    """LOESS should be somewhat less sensitive to single spikes than a global fit."""
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1
    y[10] = 50
    smoother = LoessSmoother(span=0.3, degree=1)
    y_smooth = smoother.fit_transform(y)[0]

    assert np.abs(y_smooth[0] - y[0]) < 1.0
    assert np.abs(y_smooth[-1] - y[-1]) < 1.0
