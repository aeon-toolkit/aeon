"""Tests for KNeighborsTimeSeriesRegressor."""

import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import mean_squared_error

from aeon.datasets import load_covid_3month
from aeon.distances import get_distance_function
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

distance_functions = ["euclidean", "dtw", "wdtw", "msm", "erp", "adtw", "twe"]

# expected mse on test set using default parameters.
expected_mse = {
    "euclidean": 0.002815386587822589,
    "dtw": 0.002921957478363366,
    "wdtw": 0.0025029139202436303,
    "msm": 0.002427566155284863,
    "erp": 0.002247674986547397,
    "adtw": 0.00265555172857104,
    "twe": 0.0028423024613138774,
}

# expected mse on test set using window params.
expected_mse_window = {
    "dtw": 0.0027199984296669712,
    "wdtw": 0.0026043512829531305,
    "msm": 0.002413148537646331,
    "erp": 0.0021331320891357546,
    "adtw": 0.0027602314681382163,
    "twe": 0.0030244991099088346,
}


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_neighbors(distance_key):
    """Tests kneighbors method."""
    X_train, y_train = load_covid_3month(split="train")
    X_test, y_test = load_covid_3month(split="test")

    model = KNeighborsTimeSeriesRegressor(
        n_neighbors=1, weights="distance", distance=distance_key
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    assert_almost_equal(mse, expected_mse[distance_key])


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_bounding_matrix(distance_key):
    """Test knn with custom bounding parameters, and using callables."""
    if distance_key == "euclidean" or distance_key == "squared":
        return
    X_train, y_train = load_covid_3month(split="train")
    X_test, y_test = load_covid_3month(split="test")
    distance_callable = get_distance_function(distance_key)

    knn = KNeighborsTimeSeriesRegressor(
        distance=distance_callable, distance_params={"window": 0.1}
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    assert_almost_equal(mse, expected_mse_window[distance_key])


if __name__ == "__main__":
    for distance_key in distance_functions:
        test_knn_neighbors(distance_key)
        test_knn_bounding_matrix(distance_key)
