"""Tests for the QUANTClassifier class."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC

from aeon.classification.interval_based import QUANTClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_alternative_estimator():
    """Test QUANTClassifier with an alternative estimator."""
    X, y = make_example_3d_numpy()
    clf = QUANTClassifier(estimator=RidgeClassifierCV())
    clf.fit(X, y)
    pred = clf.predict(X)

    assert isinstance(pred, np.ndarray)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_invalid_inputs():
    """Test handling of invalid inputs by QUANTClassifier."""
    X, y = make_example_3d_numpy()

    with pytest.raises(ValueError, match="quantile_divisor must be >= 1"):
        quant = QUANTClassifier(quantile_divisor=0)
        quant.fit(X, y)

    with pytest.raises(ValueError, match="interval_depth must be >= 1"):
        quant = QUANTClassifier(interval_depth=0)
        quant.fit(X, y)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_predict_proba():
    """Test predict proba with a sklearn classifier without predict proba."""
    X, y = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=12)
    r = QUANTClassifier(estimator=SVC())
    r.fit(X, y)
    p = r.predict_proba(X)
    assert p.shape == (5, 2)
