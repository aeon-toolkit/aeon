"""Tests for the LOF class."""

import numpy as np
import pytest

from aeon.anomaly_detection import LOF
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_default():
    """Test LOF with default parameters."""
    series = make_example_1d_numpy(n_timepoints=80, random_state=0)
    series[50:58] -= 2  # Introducing anomalies

    # Define common parameters
    n_neighbors = 20
    algorithm = "auto"
    leaf_size = 30
    metric = "minkowski"
    p = 2
    n_jobs = 1
    window_size = 1  # Set window_size to 1 for point-based processing
    stride = 1  # Set stride to 1

    # Initialize aeon LOF with window_size=1 and stride=1
    lof_aeon = LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        window_size=window_size,
        stride=stride,
    )
    scores_aeon = lof_aeon.fit_predict(series)

    # Ensure shapes and types
    assert scores_aeon.shape == (80,)
    assert scores_aeon.dtype == np.float64

    # Ensure that the most anomalous point is within the introduced anomaly range
    assert (
        50 <= np.argmax(scores_aeon) <= 58
    ), "AEON LOF did not detect anomalies in the expected range."


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_parameter_passing():
    """Test that LOF parameters are correctly passed to the PyOD model."""
    n_neighbors = 15
    algorithm = "kd_tree"
    leaf_size = 20
    metric = "euclidean"
    p = 1
    n_jobs = 2
    window_size = 10
    stride = 1

    # Creating LOF instance with specified parameters
    lof = LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        window_size=window_size,
        stride=stride,
    )

    # Checking if the parameters are correctly set in the PyOD model
    assert lof.pyod_model.n_neighbors == n_neighbors
    assert lof.pyod_model.algorithm == algorithm
    assert lof.pyod_model.leaf_size == leaf_size
    assert lof.pyod_model.metric == metric
    assert lof.pyod_model.p == p
    assert lof.pyod_model.n_jobs == n_jobs
    assert lof.pyod_model.contamination == 0.1  # Default contamination set internally
    assert (
        lof.pyod_model.novelty is False
    )  # novelty is set to False for unsupervised learning


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_unsupervised():
    """Test LOF in unsupervised mode (novelty=False) using decision_function."""
    series = make_example_1d_numpy(n_timepoints=80, random_state=0)
    series[50:58] -= 2

    # Define common parameters
    n_neighbors = 20
    algorithm = "auto"
    leaf_size = 30
    metric = "minkowski"
    p = 2
    novelty = False  # unsupervised learning
    n_jobs = 1
    window_size = 1  # Set window_size to 1
    stride = 1  # Set stride to 1

    # Initialize aeon LOF with window_size=1 and stride=1
    lof_aeon = LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        window_size=window_size,
        stride=stride,
    )
    scores_aeon = lof_aeon.fit_predict(series)

    # Initialize PyOD LOF
    from pyod.models.lof import LOF as PyOD_LOF

    lof_pyod = PyOD_LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        novelty=novelty,
        n_jobs=n_jobs,
    )
    lof_pyod.fit(series.reshape(-1, 1))
    scores_pyod = lof_pyod.decision_scores_

    # Ensure shapes and types
    assert scores_aeon.shape == (80,)
    assert scores_pyod.shape == (80,)
    assert scores_aeon.dtype == np.float64
    assert scores_pyod.dtype == np.float64

    # Compare anomaly scores using assert_allclose
    np.testing.assert_allclose(scores_aeon, scores_pyod, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_semi_supervised():
    """Test LOF in semi-supervised mode (novelty=True) using decision_function."""
    series_train = make_example_1d_numpy(n_timepoints=100, random_state=0)
    series_train[:10] += 5

    series_test = make_example_1d_numpy(n_timepoints=20, random_state=1)
    series_test[5:15] -= 3

    # Define common parameters
    n_neighbors = 20
    algorithm = "auto"
    leaf_size = 30
    metric = "minkowski"
    p = 2
    novelty = True  # semi-supervised learning
    n_jobs = 1
    window_size = 1  # Set window_size to 1
    stride = 1  # Set stride to 1

    # Initialize aeon LOF with window_size=1 and stride=1
    lof_aeon = LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        window_size=window_size,
        stride=stride,
    )
    lof_aeon.fit(series_train)
    scores_aeon = lof_aeon.predict(series_test)  # changed from decsion_function

    # Initialize PyOD LOF
    from pyod.models.lof import LOF as PyOD_LOF

    lof_pyod = PyOD_LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        novelty=novelty,
        n_jobs=n_jobs,
    )
    lof_pyod.fit(series_train.reshape(-1, 1))
    scores_pyod = lof_pyod.decision_function(series_test.reshape(-1, 1))

    # Ensure shapes and types
    assert scores_aeon.shape == (20,)
    assert scores_pyod.shape == (20,)
    assert scores_aeon.dtype == np.float64
    assert scores_pyod.dtype == np.float64

    # Compare anomaly scores using assert_allclose
    np.testing.assert_allclose(scores_aeon, scores_pyod, rtol=1e-5, atol=1e-5)
