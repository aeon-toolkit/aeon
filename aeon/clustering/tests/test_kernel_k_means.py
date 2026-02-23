"""Tests for time series kernel kmeans."""

import numpy as np
import pytest

from aeon.clustering._kernel_k_means import TimeSeriesKernelKMeans, _kdtw
from aeon.datasets import load_basic_motions
from aeon.utils.validation._dependencies import _check_estimator_deps

expected_labels = [0, 2, 1, 2, 0]

expected_iters = 2

expected_results = [0, 0, 0, 0, 0]

expected_labels_kdtw = [0, 0, 0, 1, 2]

expected_iters_kdtw = 2

expected_results_kdtw = [0, 2, 0, 0, 0]

max_train = 5

X_train, y_train = load_basic_motions(split="train")
X_test, y_test = load_basic_motions(split="test")


@pytest.mark.skipif(
    not _check_estimator_deps(TimeSeriesKernelKMeans, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kernel_k_means_gak():
    """Test implementation of kernel k means with GAK kernel."""
    kernel_kmeans = TimeSeriesKernelKMeans(random_state=1, n_clusters=3)
    kernel_kmeans.fit(X_train[0:max_train])
    test_shape_result = kernel_kmeans.predict(X_test[0:max_train])
    proba = kernel_kmeans.predict_proba(X_test[0:max_train])

    assert np.array_equal(test_shape_result, expected_results)
    assert kernel_kmeans.n_iter_ == expected_iters
    assert np.array_equal(kernel_kmeans.labels_, expected_labels)
    assert proba.shape == (max_train, 3)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


@pytest.mark.skipif(
    not _check_estimator_deps(TimeSeriesKernelKMeans, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kernel_k_means_kdtw():
    """Test implementation of kernel k means with KDTW kernel."""
    kernel_kmeans_kdtw = TimeSeriesKernelKMeans(
        kernel="kdtw",
        random_state=1,
        n_clusters=3,
        kernel_params={"sigma": 2.0, "epsilon": 1e-4},
    )
    kernel_kmeans_kdtw.fit(X_train[0:max_train])
    kdtw_results = kernel_kmeans_kdtw.predict(X_test[0:max_train])
    kdtw_proba = kernel_kmeans_kdtw.predict_proba(X_test[0:max_train])

    assert np.array_equal(kdtw_results, expected_results_kdtw)
    assert kernel_kmeans_kdtw.n_iter_ == expected_iters_kdtw
    assert np.array_equal(kernel_kmeans_kdtw.labels_, expected_labels_kdtw)
    assert kdtw_proba.shape == (max_train, 3)

    for val in kdtw_proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_kdtw_kernel_univariate():
    """Test kdtw kernel for univariate time series."""
    # expected value created with the original (Matlab) code from:
    # https://people.irisa.fr/Pierre-Francois.Marteau/REDK/KDTW/KDTW.html
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64).reshape(-1, 1)
    y = np.array([5, 6, 7, 8, 9, 1, 2], dtype=np.float64).reshape(-1, 1)
    sigma = 0.125
    epsilon = 1e-20
    expected_distance = 1.2814e-102

    distance = _kdtw(x, y, sigma=sigma, epsilon=epsilon)
    np.testing.assert_allclose(expected_distance, distance, rtol=1e-4, atol=1e-106)
