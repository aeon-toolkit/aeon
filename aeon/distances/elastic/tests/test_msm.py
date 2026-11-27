"""Tests for MSM distance."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import msm_cost_matrix, msm_distance, msm_pairwise_distance
from aeon.distances.elastic._msm import _cost_dependent
from aeon.testing.utils._mixed_dtypes import (
    ITAKURA,
    LONG_LENGTH,
    MIXED_CHANNELS,
    MIXED_DTYPES,
    MIXED_LENGTHS,
    N_CASES_X,
    N_CASES_Y,
    SHORT_LENGTH,
    WINDOW,
    _call_bounded,
    _make_mixed,
    _mixed_rtol,
)


def _reference_cost_dependent(x, y, z, c):
    """Previous NumPy formulation of the dependent MSM split/merge cost."""
    diameter = np.sum((y - z) ** 2)
    distance_to_mid = np.sum(((y + z) / 2.0 - x) ** 2)
    if distance_to_mid <= diameter / 4.0:
        return c
    dist_to_q_prev = np.sum((y - x) ** 2)
    dist_to_c = np.sum((z - x) ** 2)
    if dist_to_q_prev < dist_to_c:
        return c + dist_to_q_prev
    return c + dist_to_c


@pytest.mark.parametrize("n_channels", [1, 3, 10])
def test_cost_dependent_matches_reference(n_channels):
    """_cost_dependent is numerically equivalent to the previous formulation."""
    rng = np.random.default_rng(n_channels)
    for _ in range(200):
        x, y, z = rng.normal(scale=rng.uniform(0.1, 10.0), size=(3, n_channels))
        assert_allclose(
            _cost_dependent(x, y, z, 1.0),
            _reference_cost_dependent(x, y, z, 1.0),
            rtol=1e-12,
        )


def test_dependent_msm_rejects_different_channel_counts():
    """Dependent MSM raises a ValueError when channel counts differ."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20))
    y = rng.normal(size=(2, 25))
    match = "same number of channels"

    with pytest.raises(ValueError, match=match):
        msm_distance(x, y, independent=False)
    with pytest.raises(ValueError, match=match):
        msm_cost_matrix(x, y, independent=False)
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match=match):
        msm_distance(x, y, independent=False, window=0.5)


@pytest.mark.parametrize("dtype_x,dtype_y", MIXED_DTYPES)
@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("bounding", [{}, WINDOW, ITAKURA])
def test_msm_distance_mixed_dtypes(dtype_x, dtype_y, independent, bounding):
    """msm_distance on series with different dtypes equals the cost matrix corner.

    The rolling row kernels put the longer series first. Doing so by reassigning
    x and y fails Numba type unification when the dtypes differ, even when the
    series are equal length and no swap is needed.
    """
    rng = np.random.default_rng(0)
    for n_channels in MIXED_CHANNELS:
        for x_size, y_size in MIXED_LENGTHS:
            x = _make_mixed((n_channels, x_size), dtype_x, rng)
            y = _make_mixed((n_channels, y_size), dtype_y, rng)

            dist = _call_bounded(msm_distance, x, y, bounding, independent=independent)
            cost_matrix = _call_bounded(
                msm_cost_matrix, x, y, bounding, independent=independent
            )

            # An infinite distance would agree while testing nothing.
            assert np.isfinite(dist)
            assert_allclose(
                dist, cost_matrix[-1, -1], rtol=_mixed_rtol(dtype_x, dtype_y)
            )


@pytest.mark.parametrize("dtype_y", [np.int64, np.float32])
@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("bounding", [{}, WINDOW])
def test_msm_pairwise_distance_mixed_dtypes(dtype_y, independent, bounding):
    """msm_pairwise_distance on collections with different dtypes matches pairs.

    Each entry must equal the cost matrix corner for that pair of series, in
    both argument orders.
    """
    n_channels = MIXED_CHANNELS[-1]
    rng = np.random.default_rng(0)
    X = _make_mixed((N_CASES_X, n_channels, SHORT_LENGTH), np.float64, rng)
    Y = _make_mixed((N_CASES_Y, n_channels, LONG_LENGTH), dtype_y, rng)

    for a, b in [(X, Y), (Y, X)]:
        pw = _call_bounded(
            msm_pairwise_distance, a, b, bounding, independent=independent
        )
        expected = np.empty((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                expected[i, j] = _call_bounded(
                    msm_cost_matrix, a[i], b[j], bounding, independent=independent
                )[-1, -1]
        assert_allclose(pw, expected, rtol=_mixed_rtol(dtype_y))


def _reference_cost_independent(x, y, z, c):
    """Independent MSM split/merge cost, written directly from the definition."""
    if min(y, z) <= x <= max(y, z):
        return c
    return c + min(abs(x - y), abs(x - z))


def _reference_msm(x, y, c, independent):
    """Full cost matrix MSM in plain Python, no Numba and no fastmath."""
    if independent:
        return sum(
            _reference_msm_matrix(x[k : k + 1], y[k : k + 1], c, True)
            for k in range(min(len(x), len(y)))
        )
    return _reference_msm_matrix(x, y, c, False)


def _reference_msm_matrix(x, y, c, univariate):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if univariate:
        cost = lambda a, b, d: _reference_cost_independent(a[0], b[0], d[0], c)  # noqa
        dist = lambda a, b: abs(a[0] - b[0])  # noqa
    else:
        cost = lambda a, b, d: _reference_cost_dependent(a, b, d, c)  # noqa
        dist = lambda a, b: np.sum((a - b) ** 2)  # noqa
    n, m = x.shape[1], y.shape[1]
    D = np.empty((n, m))
    D[0, 0] = dist(x[:, 0], y[:, 0])
    for i in range(1, n):
        D[i, 0] = D[i - 1, 0] + cost(x[:, i], x[:, i - 1], y[:, 0])
    for j in range(1, m):
        D[0, j] = D[0, j - 1] + cost(y[:, j], x[:, 0], y[:, j - 1])
    for i in range(1, n):
        for j in range(1, m):
            D[i, j] = min(
                D[i - 1, j - 1] + dist(x[:, i], y[:, j]),
                D[i - 1, j] + cost(x[:, i], x[:, i - 1], y[:, j]),
                D[i, j - 1] + cost(y[:, j], x[:, i], y[:, j - 1]),
            )
    return D[-1, -1]


def _random_pair(rng, n_channels):
    x_size, y_size = rng.integers(1, 25, size=2)
    x = rng.normal(scale=rng.uniform(0.1, 10.0), size=(n_channels, x_size))
    y = rng.normal(scale=rng.uniform(0.1, 10.0), size=(n_channels, y_size))
    return x, y


def _tie_heavy_pairs(rng, n_channels):
    """Integer-valued inputs, so every operation is exact and ties are common."""
    for _ in range(15):
        x_size, y_size = rng.integers(1, 25, size=2)
        x = rng.integers(-2, 3, size=(n_channels, x_size)).astype(np.float64)
        y = rng.integers(-2, 3, size=(n_channels, y_size)).astype(np.float64)
        yield x, y
        # Long runs of repeated values.
        runs = rng.integers(-3, 4, size=(n_channels, x_size))
        yield np.repeat(runs, 3, axis=1).astype(np.float64), y
        # Constant series.
        yield np.full((n_channels, x_size), 2.0), y
        yield np.full((n_channels, x_size), 2.0), np.full((n_channels, y_size), -1.0)
        # Identical series, including non-integer values.
        z = rng.normal(size=(n_channels, x_size))
        yield z, z.copy()


@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_msm_distance_matches_reference(independent, n_channels):
    """Unbounded MSM matches a plain Python full cost matrix reference."""
    rng = np.random.default_rng(3562 + n_channels + 10 * independent)
    for _ in range(60):
        x, y = _random_pair(rng, n_channels)
        for c in (0.0, 0.1, 1.0, 1e6):
            assert_allclose(
                msm_distance(x, y, c=c, independent=independent),
                _reference_msm(x, y, c, independent),
                rtol=1e-12,
            )


@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_msm_distance_tie_heavy_exact(independent, n_channels):
    """On tie-heavy inputs with exact arithmetic, MSM equals the reference exactly.

    The split/merge cost is discontinuous at its boundary, so a reused or
    reordered intermediate that changed a tie would change the distance by more
    than rounding.
    """
    rng = np.random.default_rng(n_channels + 10 * independent)
    for x, y in _tie_heavy_pairs(rng, n_channels):
        for c in (0.0, 0.5, 1.0, 100.0):
            for a, b in ((x, y), (y, x)):
                assert msm_distance(
                    a, b, c=c, independent=independent
                ) == _reference_msm(a, b, c, independent)


@pytest.mark.parametrize("dtype_x,dtype_y", MIXED_DTYPES)
@pytest.mark.parametrize("independent", [True, False])
def test_msm_distance_mixed_dtypes_exact(dtype_x, dtype_y, independent):
    """Integer-valued series of mixed dtypes give exactly the reference distance."""
    rng = np.random.default_rng(0)
    for x_size, y_size in MIXED_LENGTHS:
        x = rng.integers(-3, 4, size=(3, x_size)).astype(dtype_x)
        y = rng.integers(-3, 4, size=(3, y_size)).astype(dtype_y)
        assert msm_distance(x, y, independent=independent) == _reference_msm(
            x, y, 1.0, independent
        )


@pytest.mark.parametrize("dataset", ["BasicMotions", "JapaneseVowels"])
@pytest.mark.parametrize("independent", [True, False])
def test_msm_pairwise_matches_single_calls(dataset, independent):
    """Pairwise and test x train matrices equal per-pair msm_distance exactly."""
    from aeon.datasets import load_basic_motions, load_japanese_vowels

    loader = {
        "BasicMotions": load_basic_motions,
        "JapaneseVowels": load_japanese_vowels,
    }[dataset]
    train = loader(split="TRAIN")[0][:15]
    test = loader(split="TEST")[0][:10]

    self_pw = msm_pairwise_distance(train, independent=independent)
    test_train = msm_pairwise_distance(test, train, independent=independent)
    for i in range(len(train)):
        for j in range(len(train)):
            expected = (
                0.0
                if i == j
                else msm_distance(train[i], train[j], independent=independent)
            )
            assert self_pw[i, j] == expected
    for i in range(len(test)):
        for j in range(len(train)):
            assert test_train[i, j] == msm_distance(
                test[i], train[j], independent=independent
            )
