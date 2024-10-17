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
    series[50:58] -= 2  # Introduce anomalies

    lof = LOF(window_size=10, stride=1)
    pred = lof.fit_predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_no_window():
    """Test LOF with window size of 1 (no window)."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2  # Introduce anomalies
    lof = LOF(window_size=1, stride=1)
    pred = lof.fit_predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_invalid_window_size():
    """Test LOF with invalid window size."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)

    # Expect a ValueError if window_size is set incorrectly
    with pytest.raises(ValueError):
        lof = LOF(window_size=100, stride=1)  # window size > series length
        lof.fit_predict(series)

    with pytest.raises(ValueError):
        lof = LOF(window_size=0, stride=1)  # window size < 1
        lof.fit_predict(series)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_invalid_stride():
    """Test LOF with invalid stride (greater than window size)."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)

    # Expect a ValueError if stride > window_size
    with pytest.raises(ValueError, match="Stride.*cannot be greater than window size"):
        lof = LOF(window_size=5, stride=6)  # Invalid stride
        lof.fit_predict(series)


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_multivariate():
    """Test LOF on multivariate data."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2  # Introduce anomalies in the first column

    lof = LOF(window_size=10, stride=1)
    pred = lof.fit_predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_lof_semi_supervised():
    """Test LOF in a semi-supervised setting."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2  # Introduce anomalies
    train_series = make_series(n_timepoints=100, return_numpy=True, random_state=0)

    lof = LOF(window_size=10, stride=1)
    lof.fit(train_series)
    pred = lof.predict(series)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
