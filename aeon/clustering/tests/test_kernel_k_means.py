"""Tests for time series kernel kmeans."""
import numpy as np
import pytest

from aeon.clustering._kernel_k_means import TimeSeriesKernelKMeans
from aeon.datasets import load_basic_motions
from aeon.utils.validation._dependencies import _check_estimator_deps

expected_labels = [0, 2, 1, 2, 0]

expected_score = 4.0

expected_iters = 2

expected_results = [0, 0, 0, 0, 0]


@pytest.mark.skipif(
    not _check_estimator_deps(TimeSeriesKernelKMeans, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_kernel_k_means():
    """Test implementation of kernel k means."""
    max_train = 5

    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kernel_kmeans = TimeSeriesKernelKMeans(random_state=1, n_clusters=3)
    kernel_kmeans.fit(X_train[0:max_train])
    test_shape_result = kernel_kmeans.predict(X_test[0:max_train])
    score = kernel_kmeans.score(X_test[0:max_train])
    proba = kernel_kmeans.predict_proba(X_test[0:max_train])

    assert np.array_equal(test_shape_result, expected_results)
    np.testing.assert_almost_equal(score, expected_score)
    assert kernel_kmeans.n_iter_ == expected_iters
    assert np.array_equal(kernel_kmeans.labels_, expected_labels)
    assert proba.shape == (max_train, 3)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
