"""Tests for the RSTSF class."""

import numpy as np
import pytest

from aeon.classification.interval_based import RSTSF
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency statsmodels not available",
)
def test_rstsf_string_class_labels():
    """Test RSTSF with string class labels.

    RSTSF always balances classes. String labels that parse as integers are
    mishandled by scikit-learn when given as class_weight="balanced", see
    _resolve_balanced_class_weight.
    """
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]
    y = np.asarray(y, dtype=str)

    clf = RSTSF(n_estimators=10, n_intervals=2, random_state=0)
    clf.fit(X, y)

    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))
