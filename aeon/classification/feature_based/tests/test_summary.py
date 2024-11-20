"""Test summary classifier."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from aeon.classification.feature_based import SummaryClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_summary_classifier():
    """Test the SummaryClassifier."""
    X, y = make_example_3d_numpy()
    cls = SummaryClassifier(estimator=RandomForestClassifier(n_jobs=1))
    cls.fit(X, y)
    p = cls.predict_proba(X)
    assert cls.estimator_.n_jobs == 1 and cls.n_jobs == 1
    cls = SummaryClassifier(estimator=RidgeClassifier())
    cls.fit(X, y)
    p = cls.predict_proba(X)
    assert np.all(np.isin(p, [0, 1]))
