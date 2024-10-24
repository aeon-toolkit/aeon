"""Tests for the CBLOF class."""

import numpy as np
import pytest

from aeon.anomaly_detection import CBLOF
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_cblof_default():
    """Test CBLOF."""
    series = make_example_1d_numpy(n_timepoints=80, random_state=0)
    series[50:58] -= 2

    cblof = CBLOF(window_size=10, stride=1, random_state=2)
    pred = cblof.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_cblof_pyod_parameters():
    """Test parameters are correctly passed to the CBLOF PyOD model."""
    params = {
        "n_clusters": 3,
        "alpha": 0.5,
        "beta": 2,
    }
    cblof = CBLOF(**params)

    assert cblof.pyod_model.n_clusters == params["n_clusters"]
    assert cblof.pyod_model.alpha == params["alpha"]
    assert cblof.pyod_model.beta == params["beta"]


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_aeon_copod_with_pyod_copod():
    """Test CBLOF with PyOD CBLOF."""
    from pyod.models.cblof import CBLOF as PyODCBLOF

    series = make_example_1d_numpy(n_timepoints=100, random_state=0)
    series[20:30] -= 2

    # fit and predict with aeon CBLOF
    copod = CBLOF(window_size=1, stride=1, random_state=2)
    copod_preds = copod.fit_predict(series)

    # fit and predict with PyOD CBLOF
    _series = series.reshape(-1, 1)
    pyod_copod = PyODCBLOF(random_state=2)
    pyod_copod.fit(_series)
    pyod_copod_preds = pyod_copod.decision_function(_series)

    assert np.allclose(copod_preds, pyod_copod_preds)
