"""TEASER test code."""

from sys import platform

import numpy as np
import pytest
from numpy import testing
from sklearn.ensemble import IsolationForest

from aeon.classification.early_classification._teaser import TEASER
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_unit_test


def test_teaser_with_different_decision_maker():
    """Test of TEASER with different One-Class-Classifier."""
    X_train, y_train, X_test, _, indices = load_unit_data()
    X_test = X_test[indices]

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
        one_class_classifier=IsolationForest(n_estimators=5, random_state=0),
        one_class_param_grid={"bootstrap": [True, False]},
    )
    teaser.fit(X_train[indices], y_train[indices])

    full_probas, _ = teaser.predict_proba(X_test)

    # We cannot guarantee same results on ARM macOS
    if platform != "darwin":
        testing.assert_array_almost_equal(
            full_probas, teaser_if_unit_test_probas, decimal=2
        )

    # make sure update ends up with the same probas
    teaser.reset_state_info()

    final_probas = np.zeros((10, 2))
    open_idx = np.arange(0, 10)

    for i in teaser.classification_points:
        probas, decisions = teaser.update_predict_proba(X_test[:, :, :i])
        X_test, open_idx, final_idx = teaser.split_indices_and_filter(
            X_test, open_idx, decisions
        )
        final_probas[final_idx] = probas[decisions]

        if len(X_test) == 0:
            break

    # We cannot guarantee same results on ARM macOS
    if platform != "darwin":
        testing.assert_array_almost_equal(
            final_probas, teaser_if_unit_test_probas, decimal=2
        )


def test_teaser_near_classification_points():
    """Test of TEASER with incremental time stamps outside defined class points."""
    X_train, y_train, X_test, _, indices = load_unit_data()
    X_test = X_test[indices]

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 14, 18, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=5, random_state=0),
    )
    teaser.fit(X_train, y_train)

    # use test_points that are not within list above
    test_points = [7, 11, 19, 20]

    decisions = np.zeros(len(X_test), dtype=bool)
    for i in test_points:
        X_test = X_test[np.invert(decisions)]
        X = X_test[:, :, :i]

        if i == 20:
            with pytest.raises(IndexError):
                teaser.update_predict_proba(X)
        else:
            _, decisions = teaser.update_predict(X)


def test_teaser_default():
    """Test of TEASER on the full data with the default estimator."""
    X_train, y_train, X_test, y_test, indices = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
    )
    teaser.fit(X_train, y_train)

    _, acc, earl = teaser.score(X_test[indices], y_test)

    # We cannot guarantee same results on ARM macOS
    if platform != "darwin":
        testing.assert_allclose(acc, 0.6, rtol=0.01)
        testing.assert_allclose(earl, 0.766, rtol=0.01)

        testing.assert_allclose(teaser._train_accuracy, 0.9, rtol=0.01)
        testing.assert_allclose(teaser._train_earliness, 0.733, rtol=0.01)


def load_unit_data():
    """Load unit test data."""
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
    return X_train, y_train, X_test, y_test, indices


teaser_if_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.1, 0.9],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)
