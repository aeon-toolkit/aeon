"""KGMTPClassifier tests."""

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.convolution_based._kgmtp import KGMTPClassifier
from aeon.datasets import load_italy_power_demand


def test_kgmtp_classifier_default_estimator():
    """With `estimator=None`, a StandardScaler -> RidgeClassifierCV pipeline is used.

    `predict_proba` falls back to a one-hot distribution over `predict`'s output,
    since neither step of that pipeline has its own `predict_proba`.
    """
    X = np.random.default_rng(0).random(size=(20, 1, 60))
    y = np.array([0, 1] * 10)

    clf = KGMTPClassifier(random_state=0, **KGMTPClassifier._get_test_params()).fit(
        X, y
    )

    assert isinstance(clf.estimator_, Pipeline)
    assert isinstance(clf.estimator_.steps[0][1], StandardScaler)
    assert isinstance(clf.estimator_.steps[-1][1], RidgeClassifierCV)
    assert not hasattr(clf.estimator_, "predict_proba")

    proba = clf.predict_proba(X)
    preds = clf.predict(X)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], preds)


def test_kgmtp_classifier_custom_estimator():
    """A custom `estimator` is fit directly, with no `StandardScaler` wrapping.

    Its own `predict_proba` is used too, rather than falling back to the one-hot
    approximation.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(0).random(size=(20, 1, 60))
    y = np.array([0, 1] * 10)

    clf = KGMTPClassifier(
        estimator=RandomForestClassifier(n_estimators=5, random_state=0),
        random_state=0,
        **KGMTPClassifier._get_test_params(),
    ).fit(X, y)

    assert isinstance(clf.estimator_, RandomForestClassifier)

    proba = clf.predict_proba(X)
    assert not np.all((proba == 0) | (proba == 1))


def test_kgmtp_classifier_scale_hydra_passed_through():
    """`scale_hydra` is passed straight through to `KGMTP`, for either estimator.

    Default=True reproduces the original paper's pipeline; the caller's choice is
    otherwise respected as given, with the default and a custom estimator alike.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(0).random(size=(20, 1, 60))
    y = np.array([0, 1] * 10)

    for estimator in (None, RandomForestClassifier(n_estimators=5, random_state=0)):
        for scale_hydra in (True, False):
            clf = KGMTPClassifier(
                estimator=estimator,
                scale_hydra=scale_hydra,
                random_state=0,
                **KGMTPClassifier._get_test_params(),
            ).fit(X, y)

            assert clf._transformer.scale_hydra is scale_hydra


def test_kgmtp_classifier_multivariate():
    """`KGMTPClassifier` accepts and fits on multivariate series without error."""
    X = np.random.default_rng(0).random(size=(20, 3, 60))
    y = np.array([0, 1] * 10)

    clf = KGMTPClassifier(random_state=0, **KGMTPClassifier._get_test_params()).fit(
        X, y
    )
    preds = clf.predict(X)

    assert preds.shape == (20,)
    assert len(clf._transformer.base_) == 3


def test_kgmtp_classifier_beats_majority_baseline():
    """KGMTPClassifier meaningfully outperforms a majority-class baseline."""
    X_train, y_train = load_italy_power_demand(split="train")
    X_test, y_test = load_italy_power_demand(split="test")

    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    majority_baseline = accuracy_score(y_test, np.full_like(y_test, majority_class))

    clf = KGMTPClassifier(
        n_kernels=1200, max_dilations_per_kernel=4, random_state=0
    ).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))

    assert accuracy > majority_baseline + 0.3
