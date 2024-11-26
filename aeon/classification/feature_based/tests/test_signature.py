"""Test summary classifier."""

import pytest
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.feature_based import SignatureClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("esig", severity="none"),
    reason="skip test if required soft dependency esig not available",
)
def test_signature_classifier():
    """Test the SignatureClassifier."""
    X, y = make_example_3d_numpy()
    cls = SignatureClassifier(estimator=None)
    cls._fit(X, y)
    assert isinstance(cls.pipeline.named_steps["classifier"], RandomForestClassifier)
