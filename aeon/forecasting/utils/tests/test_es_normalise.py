"""Test the ES level and seasonal normalisation functions."""

import numpy as np
import pytest

from aeon.forecasting.stats import ETS
from aeon.forecasting.utils._es_normalise import (
    es_components,
    es_denormalise,
    es_normalise_windows,
    es_seasonal_index,
    fit_es_components,
)


def _seasonal_series(n=120, m=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        (100 + 0.5 * t)
        * (1 + 0.3 * np.sin(2 * np.pi * t / m))
        * np.exp(rng.normal(0, 0.02, n))
    )


def test_states_match_ets():
    """Final level and seasonal factors match a fitted aeon ETS(M,N,M)."""
    y = _seasonal_series()
    level, season, alpha, gamma, m = fit_es_components(y, 12)
    ets = ETS(
        error_type="multiplicative",
        trend_type=None,
        seasonality_type="multiplicative",
        seasonal_period=12,
    ).fit(y)
    assert m == 12
    assert alpha == pytest.approx(ets.alpha_)
    assert gamma == pytest.approx(ets.gamma_)
    assert level[-1] == pytest.approx(ets.level_)
    # aeon stores factors by phase, the path stores them by time
    phases = (np.arange(len(y), len(y) + 12)) % 12
    np.testing.assert_allclose(season[-12:], ets.seasonality_[phases])


def test_constant_seasonal_series_normalises_to_one():
    """A noise free seasonal series normalises to ones with no smoothing."""
    pattern = np.array([1.0, 2.0, 3.0, 2.0])
    y = 10 * np.tile(pattern, 10)
    level, season = es_components(y, 4, alpha=0.0, gamma=0.0)
    X, z = es_normalise_windows(
        y, level, season, window=6, horizon=5, seasonal_period=4
    )
    np.testing.assert_allclose(X, 1.0)
    np.testing.assert_allclose(z, 1.0)
    pred = es_denormalise(1.0, level, season, len(y) - 1, 5, 4)
    assert pred == pytest.approx(y[-4])


def test_round_trip():
    """Denormalising a normalised target recovers the original value."""
    y = _seasonal_series()
    level, season, _, _, m = fit_es_components(y, 12)
    for log in (False, True):
        X, z = es_normalise_windows(y, level, season, 24, 7, m, log=log)
        origin = len(y) - 1 - 7
        assert es_denormalise(z[-1], level, season, origin, 7, m, log=log) == (
            pytest.approx(y[-1])
        )
        assert X.shape == (len(y) - 24 - 7 + 1, 24)


def test_seasonal_index_uses_known_factors():
    """Targets beyond the known factors reuse the latest factor for their phase."""
    n_season, m, origin = 132, 12, 119
    idx = es_seasonal_index(np.array([120, 131, 132, 144]), origin, n_season, m)
    np.testing.assert_array_equal(idx, [120, 131, 120, 120])
    # in sample, factors after origin + m are never used
    assert es_seasonal_index(80, 60, n_season, m) == 68


def test_short_series_drops_seasonality():
    """Series shorter than two seasons use the level only."""
    y = _seasonal_series(n=20)
    level, season, _, gamma, m = fit_es_components(y, 12)
    assert m == 1 and gamma == 0.0
    np.testing.assert_allclose(season, 1.0)


@pytest.mark.parametrize("y", [np.array([1.0, 0.0, 2.0]), np.array([1.0, -1.0, 2.0])])
def test_non_positive_raises(y):
    """The multiplicative model requires strictly positive data."""
    with pytest.raises(ValueError, match="strictly positive"):
        es_components(y, 1, 0.5, 0.0)
