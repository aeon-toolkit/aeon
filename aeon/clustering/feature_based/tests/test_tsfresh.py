"""Tests for TSFresh Clusterer."""

import numpy as np
import pytest
from sklearn import metrics

from aeon.clustering.feature_based import TSFreshClusterer
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_tsfresh_univariate():
    """Test TSFresh Clusterer with univariate data."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    tsfresh = TSFreshClusterer(
        random_state=1,
        n_clusters=2,
    )
    train_result = tsfresh.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = tsfresh.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    predict_proba = tsfresh.predict_proba(X_test)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)

    assert ari_test == 0.0
    assert ari_train == 0.02240325865580448
    assert len(predict_proba) == 20
    assert np.array_equal(
        train_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    )
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    assert train_score == 0.49473684210526314
    assert test_score == 0.4789473684210526
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_tsfresh_multivariate():
    """Test TSFresh Clusterer with multivariate data."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    tsfresh = TSFreshClusterer(
        random_state=1,
        n_clusters=2,
    )
    train_result = tsfresh.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_result)
    test_result = tsfresh.predict(X_test)
    test_score = metrics.rand_score(y_test, test_result)
    predict_proba = tsfresh.predict_proba(X_test)
    ari_test = metrics.adjusted_rand_score(y_test, test_result)
    ari_train = metrics.adjusted_rand_score(y_train, train_result)

    assert ari_test == 1
    assert ari_train == 1
    assert len(predict_proba) == 20
    assert np.array_equal(
        train_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    assert np.array_equal(
        test_result,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    assert train_score == 1.0
    assert test_score == 1.0
    assert test_result.shape == (20,)
    assert train_result.shape == (20,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_all_fc_parameters():
    """Test TSFresh Clusterer with all FC parameters."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 20

    X_train = X_train[:num_points]
    X_test = X_test[:num_points]
    fc_parameters = ["minimal", "efficient", "comprehensive"]
    for fc in fc_parameters:
        tsfresh = TSFreshClusterer(
            n_clusters=2, random_state=1, default_fc_parameters=fc
        )

        train_result = tsfresh.fit_predict(X_train)
        test_result = tsfresh.predict(X_test)
        predict_proba = tsfresh.predict_proba(X_test)
        assert len(predict_proba) == 20
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (20,)
        assert train_result.shape == (20,)
