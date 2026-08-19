"""Test TVP forecaster.

Tests include convergence properties described in Durbin & Koopman, 2012.

"""

import numpy as np

from aeon.forecasting.stats._tvp import TVP


def test_direct():
    """Test aeon TVP Forecaster equivalent to statsmodels."""
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    tvp = TVP(window=5, horizon=1, var=0.01, beta_var=0.01)
    p = tvp.forecast(expected)
    p2 = tvp.direct_forecast(expected, prediction_horizon=5)
    assert p == p2[0]


def test_static_ar1_convergence_to_ols():
    """Test TVP converges to the OLS solution for a static AR(1) process."""
    # Simulate AR(1) data with constant parameters
    rng = np.random.RandomState(0)
    true_phi = 0.6
    true_intercept = 2.0
    noise_std = 0.5
    n = 500
    y = np.zeros(n)
    # Initialize y[0] near the steady-state mean to avoid startup bias
    y[0] = true_intercept / (1 - true_phi)
    for t in range(1, n):
        y[t] = true_intercept + true_phi * y[t - 1] + rng.normal(0, noise_std)
    # Fit with beta_var=0 (no parameter drift) and observation variance = noise_var
    forecaster = TVP(window=1, horizon=1, var=noise_std**2, beta_var=0.0)
    forecaster.fit(y)
    beta_est = forecaster._beta  # [intercept, phi] estimated
    # Compute static OLS estimates for comparison
    X = np.vstack(
        [np.ones(n - 1), y[: n - 1]]
    ).T  # regress y[t] on [1, y[t-1]] for t=1..n-1
    y_resp = y[1:]
    beta_ols, *_ = np.linalg.lstsq(X, y_resp, rcond=None)
    # The TVP forecaster (with no drift) should converge to OLS estimates
    assert beta_est.shape == (2,)
    # Check that estimated parameters are close to OLS solution
    assert np.allclose(beta_est, beta_ols, atol=0.1)
    # Also check they are close to true parameters
    assert abs(beta_est[0] - true_intercept) < 0.2
    assert abs(beta_est[1] - true_phi) < 0.1


def test_tvp_adapts_to_changing_coefficient():
    """Test TVP adapts its parameters when the true AR(1) coefficient changes."""
    rng = np.random.RandomState(42)
    # Piecewise AR(1): phi changes from 0.2 to 0.8 at t=100, intercept remains 1.0
    n = 200
    phi1, phi2 = 0.2, 0.8
    intercept = 1.0
    noise_std = 0.05
    y = np.zeros(n)
    # Start near the mean of first regime
    y[0] = intercept / (1 - phi1)
    # First half (t=1 to 99) with phi1
    for t in range(1, 100):
        y[t] = intercept + phi1 * y[t - 1] + rng.normal(0, noise_std)
    # Second half (t=100 to 199) with phi2
    for t in range(100, n):
        y[t] = intercept + phi2 * y[t - 1] + rng.normal(0, noise_std)
    # Fit TVP with nonzero beta_var to allow parameter drift
    forecaster = TVP(window=1, horizon=1, var=noise_std**2, beta_var=0.1)
    forecaster.fit(y)
    beta_final = forecaster._beta
    # Compute OLS on first and second half segments for reference
    X1 = np.vstack([np.ones(99), y[:99]]).T
    y1 = y[1:100]
    beta1_ols, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    # use points 100..198 to predict 101..199
    X2 = np.vstack([np.ones(n - 101), y[100 : n - 1]]).T
    y2 = y[101:n]
    beta2_ols, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    # The final estimated phi should be much closer to phi2 than phi1
    estimated_intercept, estimated_phi = beta_final[0], beta_final[1]
    # Validate that phi coefficient increased towards phi2
    assert estimated_phi > 0.5  # moved well above the initial ~0.2
    assert abs(estimated_phi - phi2) < 0.1  # close to the new true phi
    # Validate intercept remains reasonable (around true intercept)
    assert abs(estimated_intercept - intercept) < 0.5
    # Check that final phi is closer to second-half OLS estimate than first-half
    assert abs(estimated_phi - beta2_ols[1]) < abs(estimated_phi - beta1_ols[1])
