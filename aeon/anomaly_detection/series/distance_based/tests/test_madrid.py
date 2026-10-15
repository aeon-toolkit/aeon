"""Tests for the MADRID class."""

__maintainer__ = []

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.distance_based import MADRID


def _make_series_with_anomaly(n=200, anomaly=(120, 135), seed=0):
    """Sine wave with an injected anomalous segment."""
    rng = check_random_state(seed)
    series = np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.05, n)
    series[anomaly[0] : anomaly[1]] += 3.0
    return series


def test_madrid():
    """Test MADRID output on a series with a known anomaly."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=40)
    pred = ad.fit_predict(series)

    assert pred.shape == (200,)
    assert pred.dtype == np.float64
    # anomaly scores must not be a binary {0, 1} vector
    assert not np.array_equal(np.unique(pred), [0, 1])
    # highest-scoring point falls inside the injected anomaly window
    assert 118 <= np.argmax(pred) <= 135


def test_madrid_pointwise_interval_elevation():
    """Check the detected discord's whole cover is elevated, not just its start.

    DAMP's pruning means entries away from the top discord hold unrefined
    estimates, so only the top-discord region carries a pointwise guarantee; the
    assertions reflect that contract.
    """
    anomaly = (120, 135)
    series = _make_series_with_anomaly(anomaly=anomaly)
    ad = MADRID(min_length=8, max_length=20, train_test_split=40)
    pred = ad.fit_predict(series)

    # the global maximum falls inside the injected interval
    assert anomaly[0] <= np.argmax(pred) < anomaly[1]
    # the top-scoring subsequence's cover is elevated as a block: at least
    # min_length consecutive points share the maximal score
    at_max = np.flatnonzero(pred >= pred.max() - 1e-12)
    assert len(at_max) >= 8
    assert np.all(np.diff(at_max) == 1)
    # and the discord block clearly outscores the typical background point
    scored = np.ones(len(pred), dtype=bool)
    scored[:40] = False
    assert pred[at_max].min() > np.median(pred[scored])


def test_madrid_multi_length():
    """Test MADRID aggregates a multi-length discord table of the right shape."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=24, step_size=4, train_test_split=50)
    pred = ad.fit_predict(series)

    assert pred.shape == (200,)
    # warm-up region is only reference history and is scored zero
    assert np.all(pred[:50] == 0.0)
    assert pred.max() > 0.0


def test_madrid_fractional_split():
    """Test MADRID accepts a fractional train/test split."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=0.2)
    pred = ad.fit_predict(series)

    assert pred.shape == (200,)
    assert 118 <= np.argmax(pred) <= 135


def test_madrid_incorrect_input():
    """Test MADRID with invalid parameters."""
    series = _make_series_with_anomaly()

    with pytest.raises(ValueError, match="step_size"):
        MADRID(step_size=0).fit_predict(series)
    with pytest.raises(ValueError, match="min_length must be at least 4"):
        MADRID(min_length=3).fit_predict(series)
    with pytest.raises(ValueError, match="min_length"):
        MADRID(min_length=30, max_length=20).fit_predict(series)
    with pytest.raises(ValueError, match="less than min_length"):
        MADRID(min_length=8, max_length=20).fit_predict(series[:5])
    with pytest.raises(ValueError, match="double max_length"):
        MADRID(min_length=8, max_length=150).fit_predict(series)
    with pytest.raises(ValueError, match="train_test_split"):
        MADRID(min_length=8, max_length=20, train_test_split=5).fit_predict(series)


def test_predict_discords_exposes_full_output():
    """Check predict_discords mirrors the reference implementation's output.

    predict() must be exactly the pointwise reduction of it.
    """
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=40)
    pred = ad.fit_predict(series)

    out = ad.predict_discords(series)
    expected_keys = {"lengths", "scores", "locations", "discord_table", "best_interval"}
    assert set(out) == expected_keys
    n_lengths = len(out["lengths"])
    assert out["scores"].shape == (n_lengths,)
    assert out["locations"].shape == (n_lengths,)
    assert out["discord_table"].shape == (n_lengths, 200)
    assert ((out["locations"] >= 40) & (out["locations"] < 200)).all()

    rebuilt = MADRID._to_pointwise_scores(
        out["scores"], out["locations"], out["lengths"], 200
    )
    assert np.array_equal(pred, rebuilt)


def test_pointwise_scores_only_from_identified_discords():
    """Check every nonzero score comes from an identified discord.

    Each nonzero point lies inside some per-length top discord's cover, and its
    value equals one of the identified discord scores.
    """
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=40)
    pred = ad.fit_predict(series)
    out = ad.predict_discords(series)

    nonzero = np.flatnonzero(pred)
    assert len(nonzero) > 0
    covered = np.zeros(200, dtype=bool)
    for loc, m in zip(out["locations"], out["lengths"]):
        covered[loc : loc + m] = True
    assert covered[nonzero].all()
    assert np.isin(pred[nonzero], out["scores"]).all()


def test_automatic_length_selection():
    """Check max_length=None reproduces the reference's automatic mode.

    The maximum length derives from the split (split // 20) and about 50
    candidate lengths are sampled rather than sweeping every integer.
    """
    rng = check_random_state(1)
    n = 4000
    series = np.sin(np.linspace(0, 200 * np.pi, n)) + rng.normal(0, 0.05, n)
    series[3000:3050] += 3.0
    ad = MADRID(train_test_split=2000)
    pred = ad.fit_predict(series)
    assert pred.shape == (n,)

    out = ad.predict_discords(series)
    assert out["lengths"].max() <= 2000 // 20 + 1
    assert len(out["lengths"]) <= 50
    assert len(np.unique(out["lengths"])) == len(out["lengths"])
    start, end = out["best_interval"]
    assert 2995 <= start <= 3055
    assert end - start == out["lengths"][np.argmax(out["scores"])]


def test_best_interval_consistent_with_scores():
    """Check best_interval equals the argmax discord's location and extent."""
    series = _make_series_with_anomaly()
    ad = MADRID(min_length=8, max_length=20, train_test_split=40)
    ad.fit_predict(series)
    out = ad.predict_discords(series)
    best = int(np.argmax(out["scores"]))
    start, end = out["best_interval"]
    assert start == out["locations"][best]
    assert end == start + out["lengths"][best]
