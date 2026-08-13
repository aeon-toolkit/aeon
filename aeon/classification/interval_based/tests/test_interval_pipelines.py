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


def test_interval_classifier_estimator_attribute_lifecycle():
    """Test estimator and transformer attributes are created only on fit."""
    from aeon.classification.interval_based import (
        QUANTClassifier,
        RandomIntervalClassifier,
        SupervisedIntervalClassifier,
    )
    from aeon.testing.data_generation import make_example_3d_numpy

    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20)

    for cls in [
        QUANTClassifier,
        RandomIntervalClassifier,
        SupervisedIntervalClassifier,
    ]:
        params = cls._get_test_params()
        if isinstance(params, list):
            params = params[0]

        clf = cls(**params)

        assert not hasattr(clf, "estimator_")
        assert not hasattr(clf, "_estimator")
        assert not hasattr(clf, "transformer_")
        assert not hasattr(clf, "_transformer")

        clf.fit(X, y)

        assert hasattr(clf, "estimator_")
        assert not hasattr(clf, "_estimator")
        assert hasattr(clf, "transformer_")
        assert not hasattr(clf, "_transformer")


def test_rstsf_transformers_attribute_lifecycle():
    """Test transformers and series_transformers attributes are created only on fit."""
    from aeon.classification.interval_based import RSTSF
    from aeon.testing.data_generation import make_example_3d_numpy

    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20)

    params = RSTSF._get_test_params()
    if isinstance(params, list):
        params = params[0]

    clf = RSTSF(**params)

    assert not hasattr(clf, "transformers_")
    assert not hasattr(clf, "_transformers")
    assert not hasattr(clf, "series_transformers_")
    assert not hasattr(clf, "_series_transformers")

    clf.fit(X, y)

    assert hasattr(clf, "transformers_")
    assert not hasattr(clf, "_transformers")
    assert hasattr(clf, "series_transformers_")
    assert not hasattr(clf, "_series_transformers")
