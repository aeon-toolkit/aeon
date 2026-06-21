"""Tests for the soft-MSM distance and its gradient.

Soft-MSM is a differentiable squared-cost variant of MSM with no external
reference implementation, so correctness is checked against a self-authored
hard (``gamma -> 0``) reference and a finite-difference gradient check.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import soft_msm_distance
from aeon.distances._distance import SIGNED_DISTANCES
from aeon.distances.elastic.soft._soft_msm import _soft_msm_grad_x


def _between_gate_cost(x_val, y_prev, z_other, c):
    """Return the transition cost (the between-gate is gamma-free in soft-MSM)."""
    a = x_val - y_prev
    b = x_val - z_other
    u = a * b
    g = 0.5 * (1.0 - u / np.sqrt(u * u + 1e-9))
    return c + (1.0 - g) * min(a * a, b * b)


def _hard_squared_msm(x, y, c=1.0):
    """Return the ``gamma -> 0`` limit of soft-MSM: the same DP with a hard min."""
    m, n = len(x), len(y)
    cm = np.full((m, n), np.inf)
    cm[0, 0] = (x[0] - y[0]) ** 2
    for i in range(1, m):
        cm[i, 0] = cm[i - 1, 0] + _between_gate_cost(x[i], x[i - 1], y[0], c)
    for j in range(1, n):
        cm[0, j] = cm[0, j - 1] + _between_gate_cost(y[j], y[j - 1], x[0], c)
    for i in range(1, m):
        for j in range(1, n):
            d1 = cm[i - 1, j - 1] + (x[i] - y[j]) ** 2
            d2 = cm[i - 1, j] + _between_gate_cost(x[i], x[i - 1], y[j], c)
            d3 = cm[i, j - 1] + _between_gate_cost(y[j], y[j - 1], x[i], c)
            cm[i, j] = min(d1, d2, d3)
    return cm[m - 1, n - 1]


def test_soft_msm_converges_to_hard_squared_msm():
    """As ``gamma -> 0`` soft-MSM approaches the hard squared-cost MSM DP."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(9)
        y = rng.standard_normal(11)
        soft = soft_msm_distance(x, y, gamma=1e-4, c=1.0)
        hard = _hard_squared_msm(x, y, c=1.0)
        assert_allclose(soft, hard, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_soft_msm_grad_x_matches_finite_difference(gamma):
    """Analytic soft-MSM gradient must match a central finite difference."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 8))
    y = rng.standard_normal((1, 9))

    analytic_grad, distance = _soft_msm_grad_x(x, y, 1.0, gamma)
    assert_allclose(distance, soft_msm_distance(x, y, gamma=gamma), rtol=1e-9)

    eps = 1e-6
    numerical = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        xp = x.copy()
        xm = x.copy()
        xp[0, i] += eps
        xm[0, i] -= eps
        numerical[i] = (
            soft_msm_distance(xp, y, gamma=gamma)
            - soft_msm_distance(xm, y, gamma=gamma)
        ) / (2.0 * eps)
    assert_allclose(analytic_grad, numerical, rtol=1e-4, atol=1e-4)


def test_soft_msm_multivariate_is_independent_sum():
    """Multivariate soft-MSM equals the sum of per-channel univariate values."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 10))
    y = rng.standard_normal((3, 12))
    multi = soft_msm_distance(x, y, gamma=1.0)
    per_channel = sum(soft_msm_distance(x[c], y[c], gamma=1.0) for c in range(3))
    assert_allclose(multi, per_channel, rtol=1e-9)


def test_soft_msm_is_signed_not_a_metric():
    """soft-MSM is registered as signed and need not be non-negative / zero self."""
    assert "soft_msm" in SIGNED_DISTANCES
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Self-distance is not zero for a soft distance.
    assert not np.isclose(soft_msm_distance(x, x, gamma=1.0), 0.0)


def test_soft_msm_finite_across_scales():
    """Distance stays finite across series amplitudes.

    Note: the between-gate uses a fixed ``1e-9`` floor and is therefore
    amplitude-dependent (it degenerates for very small/large series). That is a
    known limitation tracked in aeon-toolkit/aeon#3518; this test only guards
    finiteness.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal(8)
    y = rng.standard_normal(8)
    for scale in (1e-3, 1.0, 1e3):
        assert np.isfinite(soft_msm_distance(x * scale, y * scale, gamma=1.0))
