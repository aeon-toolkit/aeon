"""Tests for KNeighborsTimeSeriesRegressor."""

import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import mean_squared_error

from aeon.distances import get_distance_function
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.testing.data_generation import make_example_3d_numpy

distance_functions = ["euclidean", "dtw", "wdtw", "msm", "erp", "adtw", "twe"]

# expected mse on test set using default parameters.
expected_mse = {
    "euclidean": 0.5958635513183005,
    "dtw": 0.13862493928236033,
    "wdtw": 0.13862493928236033,
    "msm": 0.10700935790251886,
    "erp": 0.2707789569252858,
    "adtw": 0.1125922971718583,
    "twe": 0.1668928688769102,
}

# expected mse on test set using window params.
expected_mse_window = {
    "dtw": 0.19829606291787538,
    "wdtw": 0.19829606291787538,
    "msm": 0.10700935790251886,
    "erp": 0.24372655531245097,
    "adtw": 0.12166501682071837,
    "twe": 0.15932454084282624,
}

X_train, y_train = make_example_3d_numpy(regression_target=True, random_state=0)
X_test, y_test = make_example_3d_numpy(regression_target=True, random_state=2)


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_neighbors(distance_key):
    """Tests kneighbours method."""
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
    distance_callable = get_distance_function(distance_key)

    knn = KNeighborsTimeSeriesRegressor(
        distance=distance_callable, distance_params={"window": 0.1}
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    assert_almost_equal(mse, expected_mse_window[distance_key])
