"""Unit tests for classification ensemble."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest
from sklearn.covariance import log_likelihood
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from aeon.classification import DummyClassifier
from aeon.classification.compose._ensemble import ClassifierEnsemble
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.mock_estimators import MockClassifier, MockClassifierFullTags

mixed_ensemble = [
    DummyClassifier(),
    SklearnDummyClassifier(strategy="stratified"),
    DummyClassifier(strategy="uniform"),
]


@pytest.mark.parametrize(
    "classifiers",
    [
        [
            DummyClassifier(),
            DummyClassifier(strategy="stratified"),
            DummyClassifier(strategy="uniform"),
        ],
        [
            SklearnDummyClassifier(),
            SklearnDummyClassifier(strategy="stratified"),
            SklearnDummyClassifier(strategy="uniform"),
        ],
        mixed_ensemble,
    ],
)
def test_classifier_ensemble(classifiers):
    """Test the classifier ensemble."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    ensemble = ClassifierEnsemble(classifiers=classifiers, random_state=0)
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


@pytest.mark.parametrize(
    "weights",
    [
        1,
        4,
        [1, 1, 1],
        [0.5, 1, 2],
    ],
)
def test_classifier_ensemble_weights(weights):
    """Test classifier ensemble weight options."""
    X_train, y_train = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, min_cases_per_label=2
    )
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    ensemble = ClassifierEnsemble(classifiers=mixed_ensemble, weights=weights)
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


@pytest.mark.parametrize(
    "majority_vote",
    [True, False],
)
def test_classifier_ensemble_majority_vote(majority_vote):
    """Test classifier ensemble prediction options."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    ensemble = ClassifierEnsemble(
        classifiers=mixed_ensemble, majority_vote=majority_vote
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


@pytest.mark.parametrize(
    "cv",
    [2, KFold(n_splits=2)],
)
@pytest.mark.parametrize(
    "metric",
    [[None, False], [accuracy_score, False], [log_likelihood, True]],
)
def test_classifier_ensemble_learned_weights(cv, metric):
    """Test classifier pipeline with learned weights."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    y_train[9] = 1
    y_train[8] = 0
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    ensemble = ClassifierEnsemble(
        classifiers=mixed_ensemble,
        cv=cv,
        metric=metric[0],
        metric_probas=metric[1],
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


# def test_ensemble_matches_hc2():
#     """Pass"""
#     pass


def test_unequal_tag_inference():
    """Test that ClassifierEnsemble infers unequal length tag correctly."""
    X, y = make_example_3d_numpy_list(
        n_cases=10, min_n_timepoints=8, max_n_timepoints=12
    )

    c1 = MockClassifierFullTags()
    c2 = MockClassifier()

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")

    # classifiers handle unequal length
    p1 = ClassifierEnsemble(classifiers=[c1, c1, c1])
    assert p1.get_tag("capability:unequal_length")
    p1.fit(X, y)

    # test they fit even if they cannot handle unequal length
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # classifiers do not handle unequal length
    p2 = ClassifierEnsemble(classifiers=[c2, c2, c2])
    assert not p2.get_tag("capability:unequal_length")
    p2.fit(X, y)

    # any classifier does not handle unequal length
    p3 = ClassifierEnsemble(classifiers=[c1, c2, c1])
    assert not p3.get_tag("capability:unequal_length")
    p3.fit(X, y)


def test_missing_tag_inference():
    """Test that ClassifierEnsemble infers missing data tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X[5, 0, 4] = np.nan

    c1 = MockClassifierFullTags()
    c2 = MockClassifier()

    assert c1.get_tag("capability:missing_values")
    assert not c2.get_tag("capability:missing_values")

    # classifiers handle missing values
    p1 = ClassifierEnsemble(classifiers=[c1, c1, c1])
    assert p1.get_tag("capability:missing_values")
    p1.fit(X, y)

    # test they fit even if they cannot handle missing data
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # classifiers do not handle missing values
    p2 = ClassifierEnsemble(classifiers=[c2, c2, c2])
    assert not p2.get_tag("capability:missing_values")
    p2.fit(X, y)

    # any classifier does not handle missing values
    p3 = ClassifierEnsemble(classifiers=[c1, c2, c1])
    assert not p3.get_tag("capability:missing_values")
    p3.fit(X, y)


def test_multivariate_tag_inference():
    """Test that ClassifierEnsemble infers multivariate tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=12)

    c1 = MockClassifierFullTags()
    c2 = MockClassifier()

    assert c1.get_tag("capability:multivariate")
    assert not c2.get_tag("capability:multivariate")

    # classifiers handle multivariate
    p1 = ClassifierEnsemble(classifiers=[c1, c1, c1])
    assert p1.get_tag("capability:multivariate")
    p1.fit(X, y)

    # test they fit even if they cannot handle multivariate
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # classifiers do not handle multivariate
    p2 = ClassifierEnsemble(classifiers=[c2, c2, c2])
    assert not p2.get_tag("capability:multivariate")
    p2.fit(X, y)

    # any classifier does not handle multivariate
    p3 = ClassifierEnsemble(classifiers=[c1, c2, c1])
    assert not p3.get_tag("capability:multivariate")
    p3.fit(X, y)
