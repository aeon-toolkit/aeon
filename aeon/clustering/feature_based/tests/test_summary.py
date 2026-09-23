"""Tests for Summary Clusterer."""

import numpy as np

from aeon.clustering.feature_based import SummaryClusterer
from aeon.datasets import load_basic_motions, load_gunpoint


def test_all_summary_stat_uni():
    """Test Summary Clusterer with all summary stat."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 8

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    summary_stats_options = ["default", "quantiles", "bowley", "tukey"]
    for summary_stat in summary_stats_options:
        summary = SummaryClusterer(random_state=1, summary_stats=summary_stat)
        train_result = summary.fit_predict(X_train)
        test_result = summary.predict(X_test)
        predict_proba = summary.predict_proba(X_test)
        assert len(predict_proba) == 8
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (8,)
        assert train_result.shape == (8,)


def test_all_summary_stat_multi():
    """Test Summary Clusterer with all summary stat."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 8

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    summary_stats_options = ["default", "quantiles", "bowley", "tukey"]
    for summary_stat in summary_stats_options:
        summary = SummaryClusterer(random_state=1, summary_stats=summary_stat)
        train_result = summary.fit_predict(X_train)
        test_result = summary.predict(X_test)
        predict_proba = summary.predict_proba(X_test)
        assert len(predict_proba) == 8
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (8,)
        assert train_result.shape == (8,)


def test_estimator_attribute_created_on_fit():
    """Test that estimator_ is only set after fitting, not before."""
    from sklearn.cluster import KMeans

    X_train, _ = load_gunpoint(split="train")
    X_train = X_train[:10]

    model = SummaryClusterer(random_state=1)

    assert not hasattr(model, "estimator_")

    model.fit(X_train)

    assert hasattr(model, "estimator_")
    assert isinstance(model.estimator_, KMeans)
    assert not hasattr(model, "_estimator")
