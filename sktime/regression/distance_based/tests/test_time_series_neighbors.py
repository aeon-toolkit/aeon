# -*- coding: utf-8 -*-
"""Tests for KNeighborsTimeSeriesRegressor."""
import numpy as np

from sktime.regression.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesRegressor,
)


def test_knn_neighbors():
    """Tests kneighbors method."""

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=100, n_features=32, random_state=42)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = KNeighborsTimeSeriesRegressor(n_neighbors=5, weights="distance")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape
