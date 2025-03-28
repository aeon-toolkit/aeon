"""MultiRocket classifier test code."""

import numpy as np

from aeon.classification.convolution_based import MultiRocketClassifier
from aeon.datasets import load_basic_motions, load_unit_test


def test_multirocket_univariate():
    """Test of MultiRocket classifier on univariate."""
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    clf = MultiRocketClassifier(random_state=0)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    assert clf.is_fitted
    assert y_pred.shape == (X_test.shape[0],)
    assert set(y_pred).issubset(set(y_train))
    assert y_proba.shape == (X_test.shape[0], len(np.unique(y_train)))
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1)


def test_multirocket_multivariate():
    """Test of MultiRocket classifier on multivariate."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    clf = MultiRocketClassifier(random_state=0)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    assert clf.is_fitted
    assert y_pred.shape == (X_test.shape[0],)
    assert set(y_pred).issubset(set(y_train))
    assert y_proba.shape == (X_test.shape[0], len(np.unique(y_train)))
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1)
