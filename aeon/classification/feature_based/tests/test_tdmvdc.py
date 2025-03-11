"""Test TDMVDC Classifier."""

import pytest

from aeon.classification.feature_based import TDMVDCClassifier
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_tdmvdc_classifier():
    """Test the TDMVDCClassifier."""
    cls = TDMVDCClassifier()
    assert cls.k1 == 2 and cls.k2 == 2
    assert cls.n_jobs == 1
