"""Tests for KNeighborsTimeSeriesRegressor."""

import numpy as np

from aeon.regression.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesRegressor,
)


def test_knn_neighbors():
    """Tests kneighbors method."""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=9, n_features=3, random_state=42)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=3, random_state=42)

    model = KNeighborsTimeSeriesRegressor(n_neighbors=5, weights="distance")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_expected = np.array([-216.06541863, -4.54133078, -324.7624233])

    assert np.abs(y_pred - y_pred_expected).max() < 1e-6
