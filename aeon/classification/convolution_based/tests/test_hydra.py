"""Tests for configurable classifiers after the Hydra feature transforms."""

import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from aeon.classification.convolution_based import (
    HydraClassifier,
    MultiRocketHydraClassifier,
)
from aeon.testing.data_generation import make_example_3d_numpy

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="requires torch"
)


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
@pytest.mark.parametrize("n_channels", [1, 2])
def test_hydra_custom_estimator(classifier, n_channels):
    """Fit a clone of the supplied classifier on real transformed features."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=n_channels, n_timepoints=16, random_state=0
    )
    y = np.where(y == 0, "first", "second")
    estimator = KNeighborsClassifier(n_neighbors=12)
    clf = classifier(n_kernels=2, n_groups=2, estimator=estimator, random_state=0)
    clf.fit(X, y)
    assert clf.estimator is estimator
    assert clf.estimator_ is not estimator
    check_is_fitted(clf.estimator_)
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)
    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert_allclose(probabilities.sum(axis=1), 1)
    assert np.any((probabilities > 0) & (probabilities < 1))
    assert_array_equal(clf.predict(X), clf.classes_[probabilities.argmax(axis=1)])


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
def test_hydra_default_estimator_parity(classifier):
    """An explicit default ridge and the implicit default produce the same output."""
    X, y = make_example_3d_numpy(n_cases=12, n_timepoints=16, random_state=0)
    y = np.where(y == 0, -2, 7)
    kwargs = dict(n_kernels=2, n_groups=2, random_state=0, class_weight="balanced")
    default = classifier(**kwargs).fit(X, y)
    explicit = classifier(
        **kwargs,
        estimator=RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10), class_weight="balanced"
        ),
    ).fit(X, y)
    assert isinstance(default.estimator_, RidgeClassifierCV)
    assert default.estimator_.class_weight == "balanced"
    assert_array_equal(default.predict(X), explicit.predict(X))
    probabilities = default.predict_proba(X)
    expected = (default.predict(X)[:, None] == default.classes_).astype(float)
    assert_array_equal(probabilities, expected)
    assert_array_equal(explicit.predict_proba(X), expected)


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
def test_hydra_estimator_clone_and_seed(classifier):
    """Custom classifier parameters survive cloning and receive a reproducible seed."""
    X, y = make_example_3d_numpy(n_cases=12, n_timepoints=16, random_state=0)
    estimator = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=99)
    original = classifier(
        n_kernels=2,
        n_groups=2,
        estimator=estimator,
        random_state=17,
        class_weight={0: 10, 1: 1},
    )
    clone = original.clone()
    assert clone.estimator is not estimator
    assert clone.get_params()["estimator__max_depth"] == 2
    original.fit(X, y)
    clone.fit(X, y)
    assert original.estimator_.class_weight is None
    assert original.estimator_.random_state == clone.estimator_.random_state
    assert original.estimator_.random_state != 99
    assert estimator.random_state == 99
    assert_array_equal(original.predict(X), clone.predict(X))
    assert_allclose(original.predict_proba(X), clone.predict_proba(X))


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
def test_hydra_positional_parameters(classifier):
    """The estimator option does not reinterpret existing positional arguments."""
    clf = classifier(2, 4, "balanced", 1, 17)
    assert clf.n_kernels == 2
    assert clf.n_groups == 4
    assert clf.class_weight == "balanced"
    assert clf.n_jobs == 1
    assert clf.random_state == 17
    assert clf.estimator is None


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
def test_hydra_existing_fitted_attributes(classifier):
    """Predict through existing fitted attributes, including older saved models."""
    X, y = make_example_3d_numpy(n_cases=12, n_timepoints=16, random_state=0)
    clf = classifier(n_kernels=2, n_groups=2, random_state=0).fit(X, y)
    labels = clf.predict(X)
    probabilities = clf.predict_proba(X)
    del clf.estimator_
    assert_array_equal(clf.predict(X), labels)
    assert_array_equal(clf.predict_proba(X), probabilities)


class _ReverseClassifier(ClassifierMixin, BaseEstimator):
    """A classifier whose probability columns use reversed class order."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)[::-1]
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        probabilities = np.arange(1, len(self.classes_) + 1, dtype=float)
        probabilities /= probabilities.sum()
        return np.tile(probabilities, (len(X), 1))

    def predict(self, X):
        return np.repeat(self.classes_[-1], len(X))


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_hydra_probability_class_order(classifier, n_classes):
    """Probability columns follow the outer classes, even with custom ordering."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_timepoints=16, n_labels=n_classes, random_state=0
    )
    y = np.array(["class_" + str(label) for label in y])
    clf = classifier(
        n_kernels=2, n_groups=2, random_state=0, estimator=_ReverseClassifier()
    ).fit(X, y)
    assert_array_equal(clf.estimator_.classes_, clf.classes_[::-1])
    expected = np.arange(n_classes, 0, -1, dtype=float)
    expected /= expected.sum()
    assert_allclose(clf.predict_proba(X), np.tile(expected, (len(X), 1)))
    assert_array_equal(
        clf.predict(X), clf.classes_[clf.predict_proba(X).argmax(axis=1)]
    )
