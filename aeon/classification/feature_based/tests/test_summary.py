"""Test summary classifier."""

import numpy as np
import pytest
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


@pytest.mark.parametrize("class_weight", ["balanced", "balanced_subsample"])
def test_summary_classifier_with_class_weight(class_weight):
    """Test summary classifier with class weight."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=12, return_y=True, random_state=0
    )
    clf = SummaryClassifier(
        estimator=RandomForestClassifier(n_estimators=5),
        random_state=0,
        class_weight=class_weight,
    )
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))
