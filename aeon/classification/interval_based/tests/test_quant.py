"""Tests for the QUANTClassifier class."""

import pytest
from sklearn.svm import SVC

from aeon.classification.interval_based import QUANTClassifier
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.testing.utils.estimator_checks import _assert_predict_probabilities
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_alternative_estimator():
    """Test QUANTClassifier with an alternative estimator."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = QUANTClassifier(estimator=SVC())
    clf.fit(X, y)
    prob = clf.predict_proba(X)
    _assert_predict_probabilities(prob, X, n_classes=2)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_invalid_inputs():
    """Test handling of invalid inputs by QUANTClassifier."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    with pytest.raises(ValueError, match="quantile_divisor must be >= 1"):
        quant = QUANTClassifier(quantile_divisor=0)
        quant.fit(X, y)

    with pytest.raises(ValueError, match="interval_depth must be >= 1"):
        quant = QUANTClassifier(interval_depth=0)
        quant.fit(X, y)
