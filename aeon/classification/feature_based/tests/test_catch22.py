"""Test catch 22 classifier."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from aeon.classification.feature_based import Catch22Classifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_catch22():
    """Test catch22 with specific parameters."""
    X, y = make_example_3d_numpy()
    c22 = Catch22Classifier(n_jobs=1, estimator=RandomForestClassifier(n_jobs=1))
    c22.fit(X, y)
    p = c22.predict_proba(X)
    assert c22.estimator_.n_jobs == 1 and c22.n_jobs == 1
    c22 = Catch22Classifier(estimator=RidgeClassifier())
    c22.fit(X, y)
    p = c22.predict_proba(X)
    assert np.all(np.isin(p, [0, 1]))
