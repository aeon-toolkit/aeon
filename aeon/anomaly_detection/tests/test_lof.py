"""Tests for the LOF class."""

import numpy as np
import pytest

from aeon.anomaly_detection import LOF
from aeon.testing.data_generation._legacy import make_series
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_default():
    """Test LOF with default parameters."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2  # Introducing anomalies

    lof = LOF()
    pred = lof.fit_predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


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
def test_lof_compare_with_pyod_direct():
    """Test that aeon LOF with window_size=1 and stride=1 matches PyOD LOF."""
    from pyod.models.lof import LOF as PyOD_LOF

    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    # Creating aeon LOF instance with window_size=1 and stride=1
    lof_aeon = LOF(window_size=1, stride=1)
    pred_aeon = lof_aeon.fit_predict(series)

    # Directly using PyOD LOF
    lof_pyod = PyOD_LOF()
    lof_pyod.fit(series.reshape(-1, 1))
    pred_pyod = lof_pyod.decision_function(series.reshape(-1, 1))

    # Checking if the predictions match
    np.testing.assert_allclose(pred_aeon, pred_pyod, rtol=1e-5)
