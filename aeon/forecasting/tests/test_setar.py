import numpy as np

from aeon.forecasting.setar import (
    SETARForecaster,
    SETARTreeForecaster,
    SETARForestForecaster,
)


def test_setar_forecaster_basic():
    y = np.arange(20, dtype=float)

    forecaster = SETARForecaster(lags=3, threshold_lag=1)
    forecaster.fit(y)

    fh = np.array([1, 2, 3])
    y_pred = forecaster.predict(fh)

    assert y_pred.shape == (3,)
    assert np.isfinite(y_pred).all()


def test_setar_tree_forecaster_basic():
    y = [
        np.arange(20, dtype=float),
        np.arange(20, 40, dtype=float),
    ]

    forecaster = SETARTreeForecaster(lags=3, threshold_lag=1)
    forecaster.fit(y)

    fh = np.array([1, 2])
    y_pred = forecaster.predict(fh)

    assert y_pred.shape == (2,)
    assert np.isfinite(y_pred).all()


def test_setar_forest_forecaster_basic():
    y = [
        np.arange(20, dtype=float),
        np.arange(20, 40, dtype=float),
    ]

    forecaster = SETARForestForecaster(
        n_estimators=3,
        lags=3,
        threshold_lag=1,
        random_state=42,
    )
    forecaster.fit(y)

    fh = np.array([1, 2])
    y_pred = forecaster.predict(fh)

    assert y_pred.shape == (2,)
    assert np.isfinite(y_pred).all()
