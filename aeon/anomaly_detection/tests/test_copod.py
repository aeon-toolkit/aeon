"""Tests for the COPOD class."""

import numpy as np
import pytest

from aeon.anomaly_detection import COPOD
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_copod_default():
    """Test COPOD."""
    series = make_example_1d_numpy(n_timepoints=80, random_state=0)
    series[50:58] -= 2

    copod = COPOD(window_size=10, stride=1)
    pred = copod.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert np.issubdtype(pred.dtype, np.floating)
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_copod_pyod_parameters():
    """Test parameters are correctly passed to the PyOD model."""
    params = {"n_jobs": 2}
    copod = COPOD(**params)

    assert copod.pyod_model.n_jobs == params["n_jobs"]


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_aeon_copod_with_pyod_copod():
    """Test COPOD with PyOD COPOD."""
    from pyod.models.copod import COPOD as PyODCOPOD

    series = make_example_1d_numpy(n_timepoints=100, random_state=0)
    series[20:30] -= 2

    # fit and predict with aeon COPOD
    copod = COPOD(window_size=1, stride=1)
    copod_preds = copod.fit_predict(series)

    # fit and predict with PyOD COPOD
    _series = series.reshape(-1, 1)
    pyod_copod = PyODCOPOD()
    pyod_copod.fit(_series)
    pyod_copod_preds = pyod_copod.decision_function(_series)

    assert np.allclose(copod_preds, pyod_copod_preds)
