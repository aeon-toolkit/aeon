"""ProbabilityThresholdEarlyClassifier test code."""

import numpy as np
import pytest
from numpy import testing

from aeon.classification.early_classification import ProbabilityThresholdEarlyClassifier
from aeon.classification.early_classification.tests.test_teaser import load_unit_data
from aeon.classification.interval_based import TimeSeriesForestClassifier


def test_early_prob_threshold_near_classification_points():
    """Test of threshold with incremental time stamps outside defined class points."""
    X_train, y_train, X_test, _, indices = load_unit_data()
    X_test = X_test[indices]

    # train probability threshold
    pt = ProbabilityThresholdEarlyClassifier(
        random_state=0,
        classification_points=[6, 10, 14, 18, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
        probability_threshold=1,
    )
    pt.fit(X_train, y_train)

    # use test_points that are not within list above
    test_points = [7, 11, 19, 20]

    decisions = np.zeros(len(X_test), dtype=bool)
    for i in test_points:
        X_test = X_test[np.invert(decisions)]
        X = X_test[:, :, :i]

        if i == 20:
            with pytest.raises(IndexError):
                pt.update_predict_proba(X)
        else:
            _, decisions = pt.update_predict_proba(X)


def test_early_prob_threshold_score():
    """Test of threshold on the full data with the default estimator."""
    X_train, y_train, X_test, y_test, indices = load_unit_data()
    X_test = X_test[indices]
    y_test = y_test[indices]

    # train probability threshold
    pt = ProbabilityThresholdEarlyClassifier(
        random_state=0,
        classification_points=[6, 10, 14, 18, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
    )
    pt.fit(X_train, y_train)

    _, acc, earl = pt.score(X_test, y_test)
    testing.assert_allclose(acc, 0.9, rtol=0.01)
    testing.assert_allclose(earl, 0.266667, rtol=0.01)

    # make sure update ends up with the same score
    pt.reset_state_info()

    final_states = np.zeros((10, 4), dtype=int)
    open_idx = np.arange(0, 10)

    for i in pt.classification_points:
        preds, decisions = pt.update_predict(X_test[:, :, :i])
        final_states[open_idx] = pt.get_state_info()
        X_test, open_idx, final_idx = pt.split_indices_and_filter(
            X_test, open_idx, decisions
        )

        if len(X_test) == 0:
            break

    _, acc, earl = pt.compute_harmonic_mean(final_states, y_test)

    testing.assert_allclose(acc, 0.9, rtol=0.01)
    testing.assert_allclose(earl, 0.266667, rtol=0.01)
