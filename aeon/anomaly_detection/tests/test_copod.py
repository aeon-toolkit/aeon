"""Tests for the COPOD class."""

import numpy as np
import pytest

from aeon.anomaly_detection import COPOD
from aeon.testing.data_generation._legacy import make_series
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_default():
    """Test COPOD."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    copod = COPOD(window_size=10, stride=1)
    pred = copod.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_multivariate():
    """Test COPOD multivariate."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    copod = COPOD(window_size=10, stride=1)
    pred = copod.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_stride():
    """Test COPOD with stride."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2

    copod = COPOD(window_size=10, stride=1)
    pred = copod.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_multivariate_stride():
    """Test COPOD multivariate with stride."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2

    copod = COPOD(window_size=10, stride=1)
    pred = copod.fit_predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_semi_supervised_univariate():
    """Test COPOD semi-supervised univariate."""
    series = make_series(n_timepoints=80, return_numpy=True, random_state=0)
    series[50:58] -= 2
    train_series = make_series(n_timepoints=100, return_numpy=True, random_state=0)

    copod = COPOD(window_size=10, stride=1)
    copod.fit(train_series, axis=0)
    pred = copod.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60


@pytest.mark.skipif(
    not _check_soft_dependencies("pyod", severity="none"),
    reason="required soft dependency PyOD not available",
)
def test_iforest_semi_supervised_multivariate():
    """Test COPOD semi-supervised multivariate."""
    series = make_series(
        n_timepoints=80, n_columns=2, return_numpy=True, random_state=0
    )
    series[50:58, 0] -= 2
    train_series = make_series(
        n_timepoints=100, n_columns=2, return_numpy=True, random_state=0
    )

    copod = COPOD(window_size=10, stride=1)
    copod.fit(train_series, axis=0)
    pred = copod.predict(series, axis=0)

    assert pred.shape == (80,)
    assert pred.dtype == np.float_
    assert 50 <= np.argmax(pred) <= 60
