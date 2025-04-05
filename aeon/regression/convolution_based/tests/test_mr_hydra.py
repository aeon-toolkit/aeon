"""Tests for MultiRocketHydra Regressor."""

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from aeon.datasets import load_covid_3month
from aeon.regression.convolution_based import MultiRocketHydraRegressor
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["torch"], severity="none"),
    reason="pytorch soft dependency not found.",
)
def test_mr_rocket_hydra_regressor():
    """Test MultiRocketHydraRegressor on covid 3-month data."""
    X, y = load_covid_3month()

    n = int(0.8 * len(X))
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]

    reg = MultiRocketHydraRegressor(n_kernels=8, n_groups=64, random_state=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    assert y_pred.shape == y_test.shape
    mse = mean_squared_error(y_test, y_pred)
    assert np.isfinite(mse)
    assert not np.any(np.isnan(y_pred))
    assert mse < 1e6
