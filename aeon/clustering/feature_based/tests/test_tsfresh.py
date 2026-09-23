"""Tests for TSFresh Clusterer."""

import numpy as np
import pytest

from aeon.clustering.feature_based import TSFreshClusterer
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_all_fc_parameters_uni():
    """Test TSFresh Clusterer with all FC parameters."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    num_points = 5

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
        assert len(predict_proba) == 5
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (5,)
        assert train_result.shape == (5,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_all_fc_parameters_multi():
    """Test TSFresh Clusterer with all FC parameters."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    num_points = 5

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
        assert len(predict_proba) == 5
        assert not np.isnan(train_result).any()
        assert not np.isnan(test_result).any()
        assert test_result.shape == (5,)
        assert train_result.shape == (5,)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tsfresh"], severity="none"),
    reason="TSFresh soft dependency unavailable.",
)
def test_estimator_attribute_created_on_fit():
    """Test that estimator_ is only set after fitting, not before."""
    from sklearn.cluster import KMeans

    X_train, _ = load_gunpoint(split="train")
    X_train = X_train[:10]

    model = TSFreshClusterer(
        n_clusters=2, random_state=1, default_fc_parameters="minimal"
    )

    assert not hasattr(model, "estimator_")

    model.fit(X_train)

    assert hasattr(model, "estimator_")
    assert isinstance(model.estimator_, KMeans)
    assert not hasattr(model, "_estimator")
