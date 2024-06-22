"""RDST tests."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.shapelet_based import RDSTClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_predict_proba():
    """RDST tests for code not covered by standard tests."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    clf = RDSTClassifier(estimator=RandomForestClassifier(n_jobs=1))
    clf.fit(X, y)
    p = clf._predict_proba(X)
    assert p.shape == (10, 2)
    p = clf._predict(X)
    assert p.shape == (10,)
