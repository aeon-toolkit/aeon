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

    lof = LOF()
    pred = lof.fit_predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 58


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

    # Creating LOF instance with specified parameters
    lof = LOF(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        window_size=10,
        stride=1,
    )

    # Checking if the parameters are correctly set in the PyOD model
    assert lof.pyod_model.n_neighbors == n_neighbors
    assert lof.pyod_model.algorithm == algorithm
    assert lof.pyod_model.leaf_size == leaf_size
    assert lof.pyod_model.metric == metric
    assert lof.pyod_model.p == p
    assert lof.pyod_model.n_jobs == n_jobs


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_unsupervised():
    """Test LOF in unsupervised mode (novelty=False) using fit_predict."""
    series = make_example_1d_numpy(n_timepoints=80, random_state=0)
    series[50:58] -= 2

    # Initializing aeon LOF in unsupervised mode
    lof_aeon = LOF()
    pred_aeon = lof_aeon.fit_predict(series)

    # Initializing PyOD LOF in unsupervised mode
    from pyod.models.lof import LOF as PyOD_LOF
    lof_pyod = PyOD_LOF()
    lof_pyod.fit(series.reshape(-1, 1))
    pred_pyod = (lof_pyod.predict(series.reshape(-1, 1)) > 0.5).astype(int)  # Returns binary labels (outlier or inlier)

    assert pred_aeon.shape == (80,)
    assert pred_pyod.shape == (80,)
    assert pred_aeon.dtype == np.int_
    assert pred_pyod.dtype == np.int_

    # Compare binary labels
    np.testing.assert_array_equal(pred_aeon, pred_pyod)

    # Additionally, ensure that anomalies are detected in the introduced range
    assert np.any(pred_aeon[50:58] == 1)
    assert np.any(pred_pyod[50:58] == 1)


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

    # Initializing aeon LOF in semi-supervised mode
    lof_aeon = LOF()
    lof_aeon.fit(series_train)
    scores_aeon = lof_aeon._predict(series_test)

    # Initializing PyOD LOF in semi-supervised mode
    from pyod.models.lof import LOF as PyOD_LOF
    lof_pyod = PyOD_LOF(novelty=True)
    lof_pyod.fit(series_train.reshape(-1, 1))
    scores_pyod = lof_pyod.decision_function(series_test.reshape(-1, 1))

    assert scores_aeon.shape == (20,)
    assert scores_pyod.shape == (20,)
    assert scores_aeon.dtype == np.float_
    assert scores_pyod.dtype == np.float_

    # Compare anomaly scores
    np.testing.assert_allclose(scores_aeon, scores_pyod, rtol=1e-5)

    # Additionally, ensure that anomalies have higher anomaly scores
    assert np.any(scores_aeon < 0)
    assert np.any(scores_pyod < 0)

