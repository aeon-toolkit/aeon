"""Tests for MultiRocket Regressor."""

import numpy as np
from aeon.regression.convolution_based import MultiRocketRegressor
from aeon.datasets import load_covid_3month
from sklearn.metrics import mean_squared_error

def test_multirocket_regressor():
    """Test MultiRocketRegressor on covid 3-month data."""

    X, y = load_covid_3month()
    n = int(0.8 * len(X))
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    reg = MultiRocketRegressor(n_kernels=500, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    assert y_pred.shape == y_test.shape
    mse = mean_squared_error(y_test, y_pred)
    assert np.isfinite(mse)
    assert not np.any(np.isnan(y_pred))
    assert mse < 1e6