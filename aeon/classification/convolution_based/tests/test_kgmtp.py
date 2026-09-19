"""KGMTPClassifier tests."""

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based._kgmtp import KGMTPClassifier
from aeon.datasets import load_italy_power_demand


def test_kgmtp_classifier_default_estimator():
    """`predict_proba` falls back to one-hot when the estimator has none.

    With `estimator=None`, a `RidgeClassifierCV` is fit internally, and
    (since it has no `predict_proba`) `predict_proba` falls back to a
    one-hot distribution over `predict`'s output.
    """
    X = np.random.default_rng(0).random(size=(20, 1, 60))
    y = np.array([0, 1] * 10)

    clf = KGMTPClassifier(random_state=0, **KGMTPClassifier._get_test_params()).fit(
        X, y
    )

    assert isinstance(clf.estimator_, RidgeClassifierCV)
    assert not hasattr(clf.estimator_, "predict_proba")

    proba = clf.predict_proba(X)
    preds = clf.predict(X)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(clf.classes_[np.argmax(proba, axis=1)], preds)


def test_kgmtp_classifier_custom_estimator():
    """A custom estimator's own `predict_proba` is used, not the fallback.

    Rather than falling back to the one-hot approximation.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(0).random(size=(20, 1, 60))
    y = np.array([0, 1] * 10)

    clf = KGMTPClassifier(
        estimator=RandomForestClassifier(n_estimators=5, random_state=0),
        random_state=0,
        **KGMTPClassifier._get_test_params(),
    ).fit(X, y)

    proba = clf.predict_proba(X)
    assert not np.all((proba == 0) | (proba == 1))


def test_kgmtp_classifier_always_scales_hydra():
    """`KGMTPClassifier` always scales the Hydra block, regardless of `KGMTP`'s default.

    It reproduces the original paper's pipeline exactly, so unlike a
    standalone `KGMTP`, this isn't meant to be a caller-facing choice.
    """
    X = np.random.default_rng(0).random(size=(20, 1, 60))
    y = np.array([0, 1] * 10)

    clf = KGMTPClassifier(random_state=0, **KGMTPClassifier._get_test_params()).fit(
        X, y
    )

    assert clf._transformer.scale_hydra is True


def test_kgmtp_classifier_beats_majority_baseline():
    """KGMTPClassifier meaningfully outperforms a majority-class baseline.

    None of the other tests in this file check that the fitted pipeline
    actually *learns* anything -- they only check API mechanics (shapes,
    the one-hot `predict_proba` fallback, etc.) and would still pass even
    if the features carried no signal at all. This is a lightweight
    sanity/regression check rather than a unit test: it fits on a small,
    fast, real UCR dataset (bundled with aeon, so no network access is
    needed) and checks accuracy clearly beats always predicting the
    training set's majority class. The margin is a threshold rather than
    a pinned exact value, so it stays robust to minor Numba/platform
    floating-point differences while still catching a badly broken
    pipeline (e.g. features carrying no signal, or labels misaligned).
    """
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
