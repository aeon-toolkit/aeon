"""Test interval pipelines."""

import pytest
from sklearn.svm import SVC

from aeon.classification.interval_based import (
    RandomIntervalClassifier,
    SupervisedIntervalClassifier,
)
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.testing.utils.estimator_checks import _assert_predict_probabilities


@pytest.mark.parametrize(
    "cls", [SupervisedIntervalClassifier, RandomIntervalClassifier]
)
def test_interval_pipeline_classifiers(cls):
    """Test the random interval classifiers."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]
    X_test, y_test = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["test"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"estimator": SVC()})

    clf = cls(**params)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)
    _assert_predict_probabilities(prob, X_test, n_classes=2)
