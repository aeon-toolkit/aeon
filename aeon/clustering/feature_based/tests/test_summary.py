"""Tests for Summary Clusterer."""

import numpy as np
from sklearn import metrics

from aeon.clustering.feature_based import SummaryClusterer
from aeon.datasets import load_basic_motions, load_gunpoint


def test_summary_univariate():
    """Test Summary Clusterer with univariate data."""

    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    summary = SummaryClusterer(
        random_state=1,
    )
    train_result = summary.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = summary.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    predict_proba = summary.predict_proba(X_test)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)

    assert ari_test == 0.026750142287990893
    assert ari_train == 0.18636519355943817
    assert len(predict_proba) == 20
    assert np.array_equal(
        test_result,
        [2, 0, 4, 2, 6, 0, 1, 7, 3, 0, 6, 2, 0, 2, 2, 6, 0, 6, 2, 6],
    )
    assert np.array_equal(
        train_result,
        [6, 6, 3, 6, 0, 6, 5, 4, 1, 2, 2, 2, 1, 2, 0, 2, 3, 6, 2, 7],
    )
    assert train_score == 0.6052631578947368
    assert test_score == 0.5263157894736842
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


def test_summary_multivariate():
    """Test Summary Clusterer with multivariate data."""

    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    summary = SummaryClusterer(
        random_state=1,
    )
    train_result = summary.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = summary.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    predict_proba = summary.predict_proba(X_test)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)

    assert ari_test == 0.43478260869565216
    assert ari_train == 0.4683038263849229
    assert len(predict_proba) == 20
    assert np.array_equal(
        test_result,
        [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 1, 5, 3, 5, 4, 4, 1, 1, 4],
    )
    assert np.array_equal(
        train_result,
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 6, 4, 7, 4, 5, 3, 1],
    )
    assert train_score == 0.7421052631578947
    assert test_score == 0.7263157894736842
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


def test_all_summary_stat():
    """Test Summary Clusterer with all summary stat."""

    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    summary_stats_options = ["default", "percentiles", "bowley", "tukey"]
    for summary_stat in summary_stats_options:
        summary = SummaryClusterer(random_state=1, summary_stats=summary_stat)
        train_result = summary.fit_predict(X_train)
        test_result = summary.predict(X_test)
        predict_proba = summary.predict_proba(X_test)
        assert len(predict_proba) == 20
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (20,)
        assert train_result.shape == (20,)
