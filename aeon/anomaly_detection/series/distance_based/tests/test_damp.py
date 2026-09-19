"""Tests for the DAMP class."""

__maintainer__ = []

import warnings

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.distance_based import DAMP
from aeon.anomaly_detection.series.distance_based._madrid import _damp, _next_pow2
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


def _make_series_with_anomaly(n=600, anomaly=(400, 415), seed=0):
    """Sine wave with an injected anomalous segment."""
    rng = check_random_state(seed)
    series = np.sin(np.linspace(0, 24 * np.pi, n)) + rng.normal(0, 0.05, n)
    series[anomaly[0] : anomaly[1]] += 3.0
    return series


def _make_series_with_anomalies(kind, n, n_anomalies, seed):
    """Random walk or noisy sine with several injected anomalies of varied shape."""
    rng = check_random_state(seed)
    if kind == "walk":
        series = np.cumsum(rng.normal(0, 1, n))
    else:
        series = np.sin(np.linspace(0, n / 8, n)) + rng.normal(0, 0.1, n)
    starts = np.sort(rng.choice(np.arange(n // 3, n - 30, 40), n_anomalies, False))
    for a, start in enumerate(starts):
        length = rng.randint(5, 20)
        segment = slice(start, start + length)
        if a % 3 == 0:
            series[segment] += rng.choice([-1, 1]) * rng.uniform(2, 5)
        elif a % 3 == 1:
            series[segment] += rng.normal(0, 2, length)
        else:
            series[segment] = series[start] + np.linspace(0, 3, length)
    return series


def _brute_force_left_matrix_profile(X, m, split):
    """Exact left matrix profile using the z-normalised distance of ``_mass``."""
    windows = np.lib.stride_tricks.sliding_window_view(X, m)
    means = windows.mean(axis=1)
    stds = np.maximum(windows.std(axis=1), AEON_NUMBA_STD_THRESHOLD)
    dots = windows @ windows.T
    left_mp = np.zeros(len(windows))
    for i in range(split, len(windows)):
        j = np.arange(i - m + 1)
        corr = (dots[i, j] - m * means[i] * means[j]) / (stds[i] * stds[j])
        left_mp[i] = np.sqrt(max(np.min(2.0 * (m - corr)), 0.0))
    return left_mp


def _brute_force_top_k(left_mp, split, k, radius):
    """Greedy top-k: take the maximum, exclude positions within radius, repeat."""
    values = left_mp.copy()
    values[:split] = -np.inf
    scores, locations = [], []
    for _ in range(k):
        loc = int(np.argmax(values))
        if values[loc] == -np.inf:
            break
        scores.append(values[loc])
        locations.append(loc)
        values[max(loc - radius, 0) : loc + radius + 1] = -np.inf
    return np.array(scores), np.array(locations, dtype=np.int64)


def test_damp():
    """Test DAMP flags the injected interval with elevated pointwise scores."""
    anomaly = (400, 415)
    series = _make_series_with_anomaly(anomaly=anomaly)
    ad = DAMP(window_size=32, n_discords=3, train_test_split=200)
    pred = ad.fit_predict(series)

    assert pred.shape == (600,)
    assert pred.dtype == np.float64
    assert not np.array_equal(np.unique(pred), [0, 1])
    assert np.all(pred[:200] == 0.0)
    # the highest score is attained inside the injected interval and every point
    # of the interval is elevated
    assert pred[anomaly[0] : anomaly[1]].max() == pred.max()
    assert np.all(pred[anomaly[0] : anomaly[1]] > 0.0)
    # elevated points only occur within one window of the injected interval
    nonzero = np.flatnonzero(pred)
    assert nonzero.min() > anomaly[0] - 32
    assert nonzero.max() < anomaly[1] + 32
    # every discord overlaps the injected interval
    locations = ad.predict_discords(series)["locations"]
    assert np.all((locations < anomaly[1]) & (locations + 32 > anomaly[0]))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["walk", "sine"])
@pytest.mark.parametrize("window_size", [8, 13, 24])
def test_damp_matches_brute_force_topk(seed, kind, window_size):
    """Test the top-k discords equal those of the exact left matrix profile."""
    n = 360
    series = _make_series_with_anomalies(kind, n, seed % 3 + 1, seed)
    split = 4 * window_size
    left_mp = _brute_force_left_matrix_profile(series, window_size, split)

    for n_discords in (1, 2, 3):
        expected_scores, expected_locations = _brute_force_top_k(
            left_mp, split, n_discords, window_size // 2
        )
        for lookahead in (0, None, window_size + 1, 3):
            ad = DAMP(
                window_size=window_size,
                n_discords=n_discords,
                train_test_split=split,
                lookahead=lookahead,
            )
            out = ad.predict_discords(series)
            np.testing.assert_allclose(out["scores"], expected_scores, rtol=1e-7)
            np.testing.assert_array_equal(out["locations"], expected_locations)


def test_damp_returns_fewer_discords_on_short_test_region():
    """Test fewer discords are returned when the test region cannot hold them."""
    series = _make_series_with_anomalies("walk", 120, 1, 3)
    window_size, split = 16, 96
    left_mp = _brute_force_left_matrix_profile(series, window_size, split)
    expected_scores, expected_locations = _brute_force_top_k(
        left_mp, split, 5, window_size // 2
    )
    out = DAMP(
        window_size=window_size, n_discords=5, train_test_split=split
    ).predict_discords(series)

    assert len(out["scores"]) < 5
    np.testing.assert_allclose(out["scores"], expected_scores, rtol=1e-7)
    np.testing.assert_array_equal(out["locations"], expected_locations)


@pytest.mark.parametrize("lookahead", [None, 0, 8])
def test_exact_repeats_keep_zero_score_discords(lookahead):
    """Test forward pruning never drops discords with a zero score.

    Every subsequence of an exactly repeating series has a left matrix profile
    value of zero, so all of them tie and forward pruning must not remove any
    before the requested number of discords is found.
    """
    series = np.tile([0.0, 1.0, 2.0], 60)
    window_size, split = 6, 24
    left_mp = _brute_force_left_matrix_profile(series, window_size, split)
    expected_scores, expected_locations = _brute_force_top_k(
        left_mp, split, 5, window_size // 2
    )
    out = DAMP(
        window_size=window_size,
        n_discords=5,
        train_test_split=split,
        lookahead=lookahead,
    ).predict_discords(series)

    assert len(expected_locations) == 5
    np.testing.assert_allclose(out["scores"], expected_scores, rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(out["locations"], expected_locations)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("window_size", [8, 16, 30])
@pytest.mark.parametrize("periodic", [False, True])
def test_damp_k1_matches_madrid_damp_kernel(seed, window_size, periodic):
    """Test n_discords=1 reproduces MADRID's single-length DAMP kernel exactly.

    The periodic case repeats a pattern exactly, so forward distances equal to the
    threshold occur and the pruning rule has to match MADRID's on ties as well.
    """
    if periodic:
        rng = check_random_state(seed)
        series = np.tile(rng.normal(0, 1, 16), 32)
        series[350:360] += rng.normal(0, 2, 10)
    else:
        kind = "sine" if seed % 2 else "walk"
        series = _make_series_with_anomalies(kind, 500, 2, seed)
    n = len(series)
    split = 5 * window_size
    best_so_far, left_mp = _damp(
        series, window_size, split, np.zeros(n - window_size + 1), 0.0
    )

    ad = DAMP(
        window_size=window_size,
        train_test_split=split,
        lookahead=2 ** _next_pow2(window_size),
    )
    out = ad.predict_discords(series)
    assert best_so_far > 0.0
    assert np.array_equal(out["left_matrix_profile"], left_mp)
    assert out["scores"][0] == best_so_far
    assert out["locations"][0] == np.argmax(left_mp)


def test_pointwise_scores_only_from_identified_discords():
    """Test every nonzero score comes from an identified discord.

    Each nonzero point lies inside some discord's cover, carries the largest
    score of the discords covering it, and predict is exactly that reduction.
    """
    series = _make_series_with_anomaly()
    ad = DAMP(window_size=32, n_discords=3, train_test_split=200)
    pred = ad.fit_predict(series)
    out = ad.predict_discords(series)

    expected = np.zeros(600)
    for score, loc in zip(out["scores"], out["locations"]):
        expected[loc : loc + 32] = np.maximum(expected[loc : loc + 32], score)
    assert np.array_equal(pred, expected)
    nonzero = np.flatnonzero(pred)
    assert len(nonzero) > 0
    assert np.isin(pred[nonzero], out["scores"]).all()
    assert np.array_equal(
        pred, DAMP._to_pointwise_scores(out["scores"], out["locations"], 32, 600)
    )


def test_predict_discords_outputs():
    """Test the keys, types and ordering of predict_discords."""
    series = _make_series_with_anomaly()
    ad = DAMP(window_size=32, n_discords=3, train_test_split=200)
    out = ad.predict_discords(series)

    expected_keys = {
        "scores",
        "locations",
        "left_matrix_profile",
        "pruning_rate",
        "best_interval",
    }
    assert set(out) == expected_keys
    assert out["scores"].dtype == np.float64
    assert out["locations"].dtype == np.int64
    assert out["scores"].shape == out["locations"].shape == (3,)
    assert np.all(np.diff(out["scores"]) <= 0)
    assert np.all((out["locations"] >= 200) & (out["locations"] <= 600 - 32))
    gaps = np.abs(out["locations"][:, None] - out["locations"][None, :])
    assert np.all(gaps[~np.eye(3, dtype=bool)] > 32 // 2)

    left_mp = out["left_matrix_profile"]
    assert left_mp.shape == (600 - 32 + 1,)
    assert np.all(left_mp[:200] == 0.0)
    # the discords carry their exact left matrix profile values
    np.testing.assert_array_equal(left_mp[out["locations"]], out["scores"])
    assert 0.0 < out["pruning_rate"] < 1.0
    assert isinstance(out["pruning_rate"], float)

    start, end = out["best_interval"]
    assert start == out["locations"][0]
    assert end == start + 32


def test_fractional_split():
    """Test DAMP accepts a fractional train/test split."""
    series = _make_series_with_anomaly()
    ad_fraction = DAMP(window_size=32, train_test_split=0.25)
    ad_index = DAMP(window_size=32, train_test_split=150)
    pred = ad_fraction.fit_predict(series)

    assert pred.shape == (600,)
    assert np.all(pred[:150] == 0.0)
    assert np.array_equal(pred, ad_index.fit_predict(series))
    assert 400 - 32 < np.argmax(pred) < 415


def test_lookahead_zero_matches_default():
    """Test forward pruning changes the speed but not the discords found."""
    series = _make_series_with_anomalies("walk", 800, 3, 7)
    for n_discords in (1, 3):
        online = DAMP(
            window_size=20, n_discords=n_discords, train_test_split=160, lookahead=0
        ).predict_discords(series)
        default = DAMP(
            window_size=20, n_discords=n_discords, train_test_split=160
        ).predict_discords(series)

        assert online["pruning_rate"] == 0.0
        assert default["pruning_rate"] > 0.0
        np.testing.assert_array_equal(online["locations"], default["locations"])
        np.testing.assert_array_equal(online["scores"], default["scores"])


def test_damp_incorrect_input():
    """Test DAMP with invalid parameters."""
    series = _make_series_with_anomaly()

    with pytest.raises(ValueError, match="window_size must be at least 4"):
        DAMP(window_size=3).fit_predict(series)
    with pytest.raises(ValueError, match="n_discords"):
        DAMP(n_discords=0).fit_predict(series)
    with pytest.raises(ValueError, match="lookahead"):
        DAMP(lookahead=-1).fit_predict(series)
    with pytest.raises(ValueError, match="double window_size"):
        DAMP(window_size=32).fit_predict(series[:60])
    with pytest.raises(ValueError, match="train_test_split"):
        DAMP(window_size=32, train_test_split=20).fit_predict(series)
    with pytest.raises(ValueError, match="train_test_split"):
        DAMP(window_size=32, train_test_split=590).fit_predict(series)
    with pytest.raises(ValueError, match="univariate"):
        DAMP(window_size=32).predict_discords(np.vstack([series, series]))


def test_warns_on_short_warm_up():
    """Test DAMP warns when the warm-up region is shorter than 4 * window_size."""
    series = _make_series_with_anomaly()
    with pytest.warns(UserWarning, match="four times window_size"):
        DAMP(window_size=32, train_test_split=64).fit_predict(series)

    # the default split uses a warm-up of max(4 * window_size, len(X) // 5) points
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pred = DAMP(window_size=32).fit_predict(series)
    assert np.all(pred[:128] == 0.0)
    assert np.array_equal(
        pred, DAMP(window_size=32, train_test_split=128).fit_predict(series)
    )


def test_warns_on_near_constant_region():
    """Test DAMP warns when a window is close to constant."""
    series = _make_series_with_anomaly()
    series[300:340] = 1.0
    with pytest.warns(UserWarning, match="close to constant"):
        DAMP(window_size=32, train_test_split=200).fit_predict(series)
