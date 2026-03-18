"""Tests for KNeighborsTimeSeriesRegressor."""

import numpy as np
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


@pytest.mark.parametrize("distance_key", distance_functions)
def test_knn_kneighbors(distance_key):
    """Test knn kneighbors with comprehensive validation."""
    knn = KNeighborsTimeSeriesRegressor(distance=distance_key, n_neighbors=3)
    knn.fit(X_train, y_train)

    # Test basic kneighbors functionality
    dists, ind = knn.kneighbors(X_test, n_neighbors=3)
    assert isinstance(dists, np.ndarray)
    assert isinstance(ind, np.ndarray)
    assert dists.shape == (X_test.shape[0], 3)
    assert ind.shape == (X_test.shape[0], 3)

    # Test that distances are non-negative
    assert np.all(dists >= 0)

    # Test that indices are within valid range
    assert np.all(ind >= 0)
    assert np.all(ind < len(X_train))

    # Test that distances are sorted (closest first)
    assert np.all(dists[:, 0] <= dists[:, 1])
    assert np.all(dists[:, 1] <= dists[:, 2])

    # Test using kneighbors results to make predictions manually
    # This validates that the kneighbors method returns correct neighbor indices
    manual_preds = np.empty(len(X_test))
    for i in range(len(X_test)):
        # Get the first neighbor (closest) for each test point
        neighbor_idx = ind[i, 0]
        manual_preds[i] = y_train[neighbor_idx]

    # Calculate MSE using manual predictions
    manual_mse = mean_squared_error(y_test, manual_preds)

    # The manual MSE should be close to the expected MSE for n_neighbors=1
    # We use a tolerance since we're only using the first neighbor
    assert_almost_equal(manual_mse, expected_mse[distance_key], decimal=1)

    # Test kneighbors with different n_neighbors values
    dists_2, ind_2 = knn.kneighbors(X_test, n_neighbors=2)
    assert dists_2.shape == (X_test.shape[0], 2)
    assert ind_2.shape == (X_test.shape[0], 2)

    # Test kneighbors without returning distances
    ind_only = knn.kneighbors(X_test, n_neighbors=3, return_distance=False)
    assert isinstance(ind_only, np.ndarray)
    assert ind_only.shape == (X_test.shape[0], 3)
    # Should return same indices as when return_distance=True
    np.testing.assert_array_equal(ind_only, ind)

    # Test kneighbors on training data (should exclude self)
    train_dists, train_ind = knn.kneighbors(n_neighbors=2)
    assert train_dists.shape == (len(X_train), 2)
    assert train_ind.shape == (len(X_train), 2)
    # Each point should not be its own neighbor (diagonal should be excluded)
    for i in range(len(X_train)):
        assert i not in train_ind[i]
