"""Tests for the PyODAdapter class."""

__maintainer__ = ["CodeLionX"]

import numpy as np
import pytest

from aeon.anomaly_detection import PyODAdapter
from aeon.testing.data_generation._legacy import make_series
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_default():
    """Test PyODAdapter."""
    from pyod.models.lof import LOF

    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    ad = PyODAdapter(LOF(), window_size=10, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
    assert ad.pyod_model.decision_scores_.shape == (71,)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_multivariate():
    """Test PyODAdapter multivariate."""
    from pyod.models.lof import LOF

    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    ad = PyODAdapter(LOF(), window_size=10, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
    assert ad.pyod_model.decision_scores_.shape == (71,)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_no_window_univariate():
    """Test PyODAdapter without windows univariate."""
    from pyod.models.lof import LOF

    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    ad = PyODAdapter(LOF(), window_size=1, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
    assert ad.pyod_model.decision_scores_.shape == (80,)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_no_window_multivariate():
    """Test PyODAdapter without windows multivariate."""
    from pyod.models.lof import LOF

    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    ad = PyODAdapter(LOF(), window_size=1, stride=1)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
    assert ad.pyod_model.decision_scores_.shape == (80,)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_stride_univariate():
    """Test PyODAdapter with stride != 1 univariate."""
    from pyod.models.lof import LOF

    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    ad = PyODAdapter(LOF(), window_size=10, stride=5)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
    assert ad.pyod_model.decision_scores_.shape == (15,)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_pyod_adapter_stride_multivariate():
    """Test PyODAdapter with stride != 1 multivariate."""
    from pyod.models.lof import LOF

    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    ad = PyODAdapter(LOF(), window_size=10, stride=5)
    pred = ad.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
    assert hasattr(ad, "pyod_model")
    assert ad.pyod_model.decision_scores_.shape == (15,)
