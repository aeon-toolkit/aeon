"""Test interval pipelines."""

import pytest
from sklearn.svm import SVC

from aeon.classification.interval_based import (
    RandomIntervalClassifier,
    SupervisedIntervalClassifier,
)
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize(
    "cls", [SupervisedIntervalClassifier, RandomIntervalClassifier]
)
def test_random_interval_classifier(cls):
    """Test the random interval classifiers."""
    X, y = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=12)
    r = cls(estimator=SVC())
    r.fit(X, y)
    p = r.predict_proba(X)
    assert p.shape == (5, 2)
    r = cls(n_jobs=2)
    r.fit(X, y)
    assert r._estimator.n_jobs == 2


def test_parameter_sets():
    """Test results comparison parameter sets."""
    paras = SupervisedIntervalClassifier._get_test_params(
        parameter_set="results_comparison"
    )
    assert paras["n_intervals"] == 2
    paras = RandomIntervalClassifier._get_test_params(
        parameter_set="results_comparison"
    )
    assert paras["n_intervals"] == 3
