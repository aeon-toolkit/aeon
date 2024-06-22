"""STC specific tests."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_predict_proba():
    """Test predict_proba when classifier has no predict_proba method."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    stc = ShapeletTransformClassifier(estimator=SVC(probability=False))
    stc.fit(X, y)
    probas = stc._predict_proba(X)
    assert np.all(
        (probas == 0.0) | (probas == 1.0)
    ), "Array contains values other than 0 and 1"
    with pytest.raises(ValueError, match="Estimator must have a predict_proba method"):
        stc._fit_predict_proba(X, y)
    stc = ShapeletTransformClassifier(estimator=RandomForestClassifier(n_estimators=10))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="All classes must have at least 2 values"):
        stc._fit_predict_proba(X, y)
