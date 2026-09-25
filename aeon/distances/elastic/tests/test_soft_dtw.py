"""Tests for the soft-DTW distance and its gradient."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.distances import soft_dtw_distance
from aeon.distances.elastic.soft._soft_dtw import _soft_dtw_grad_x


@pytest.mark.parametrize("gamma", [0.1, 1.0, 2.0])
def test_soft_dtw_matches_tslearn(gamma):
    """Soft-DTW must match the tslearn reference exactly, including negatives.

    This is the external oracle for the signed soft-DTW value. The old aeon
    implementation returned ``abs(...)``, which disagreed with tslearn whenever
    the true soft-DTW value was negative (large ``gamma``); the signed value
    matches tslearn for all ``gamma``.
    """
    tslearn_metrics = pytest.importorskip("tslearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(10)
        y = rng.standard_normal(12)
        aeon_value = soft_dtw_distance(x, y, gamma=gamma)
        tslearn_value = tslearn_metrics.soft_dtw(
            x.reshape(-1, 1), y.reshape(-1, 1), gamma=gamma
        )
        assert_allclose(aeon_value, tslearn_value, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_soft_dtw_grad_x_matches_finite_difference(gamma):
    """The analytic soft-DTW gradient must match a central finite difference.

    ``_soft_dtw_grad_x`` is private plumbing (consumed by the soft barycentre
    averaging), so correctness is verified here rather than through public API.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 8))
    y = rng.standard_normal((1, 9))

    analytic_grad, distance = _soft_dtw_grad_x(x, y, gamma)

    # The returned distance must equal the public distance on the same inputs.
    assert_allclose(distance, soft_dtw_distance(x, y, gamma=gamma), rtol=1e-9)

    eps = 1e-6
    numerical_grad = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[0, i] += eps
        x_minus[0, i] -= eps
        f_plus = soft_dtw_distance(x_plus, y, gamma=gamma)
        f_minus = soft_dtw_distance(x_minus, y, gamma=gamma)
        numerical_grad[0, i] = (f_plus - f_minus) / (2.0 * eps)

    assert_allclose(analytic_grad, numerical_grad, rtol=1e-4, atol=1e-4)
