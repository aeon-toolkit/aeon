"""Unit tests for regression ensemble."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from aeon.regression import DummyRegressor
from aeon.regression.compose._ensemble import RegressorEnsemble
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.mock_estimators import MockRegressor, MockRegressorFullTags

mixed_ensemble = [
    DummyRegressor(),
    SklearnDummyRegressor(strategy="median"),
    DummyRegressor(strategy="quantile", quantile=0.75),
]


@pytest.mark.parametrize(
    "regressors",
    [
        [
            DummyRegressor(),
            DummyRegressor(strategy="median"),
            DummyRegressor(strategy="quantile", quantile=0.75),
        ],
        [
            SklearnDummyRegressor(),
            SklearnDummyRegressor(strategy="median"),
            SklearnDummyRegressor(strategy="quantile", quantile=0.75),
        ],
        mixed_ensemble,
    ],
)
def test_regressor_ensemble(regressors):
    """Test the regressor ensemble."""
    X_train, y_train = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, regression_target=True
    )
    X_test, _ = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, regression_target=True
    )

    ensemble = RegressorEnsemble(regressors=regressors, random_state=0)
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


@pytest.mark.parametrize(
    "weights",
    [
        1,
        4,
        [1, 1, 1],
        [0.5, 1, 2],
    ],
)
def test_regressor_ensemble_weights(weights):
    """Test regressor ensemble weight options."""
    X_train, y_train = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, regression_target=True
    )
    X_test, _ = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, regression_target=True
    )

    ensemble = RegressorEnsemble(regressors=mixed_ensemble, weights=weights)
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


@pytest.mark.parametrize(
    "cv",
    [2, KFold(n_splits=2)],
)
@pytest.mark.parametrize(
    "metric",
    [None, mean_squared_error, mean_absolute_error],
)
def test_regressor_ensemble_learned_weights(cv, metric):
    """Test regressor pipeline with learned weights."""
    X_train, y_train = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, regression_target=True
    )
    X_test, _ = make_example_3d_numpy(
        n_cases=10, n_timepoints=12, regression_target=True
    )

    ensemble = RegressorEnsemble(
        regressors=mixed_ensemble,
        cv=cv,
        metric=metric,
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    assert isinstance(y_pred, np.ndarray)


def test_unequal_tag_inference():
    """Test that RegressorEnsemble infers unequal length tag correctly."""
    X, y = make_example_3d_numpy_list(
        n_cases=10, min_n_timepoints=8, max_n_timepoints=12, regression_target=True
    )

    r1 = MockRegressorFullTags()
    r2 = MockRegressor()

    assert r1.get_tag("capability:unequal_length")
    assert not r2.get_tag("capability:unequal_length")

    # regressors handle unequal length
    p1 = RegressorEnsemble(regressors=[r1, r1, r1])
    assert p1.get_tag("capability:unequal_length")
    p1.fit(X, y)

    # test they fit even if they cannot handle unequal length
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)

    # regressors do not handle unequal length
    p2 = RegressorEnsemble(regressors=[r2, r2, r2])
    assert not p2.get_tag("capability:unequal_length")
    p2.fit(X, y)

    # any regressor does not handle unequal length
    p3 = RegressorEnsemble(regressors=[r1, r2, r1])
    assert not p3.get_tag("capability:unequal_length")
    p3.fit(X, y)


def test_missing_tag_inference():
    """Test that RegressorEnsemble infers missing data tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)
    X[5, 0, 4] = np.nan

    r1 = MockRegressorFullTags()
    r2 = MockRegressor()

    assert r1.get_tag("capability:missing_values")
    assert not r2.get_tag("capability:missing_values")

    # regressors handle missing values
    p1 = RegressorEnsemble(regressors=[r1, r1, r1])
    assert p1.get_tag("capability:missing_values")
    p1.fit(X, y)

    # test they fit even if they cannot handle missing data
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)

    # regressors do not handle missing values
    p2 = RegressorEnsemble(regressors=[r2, r2, r2])
    assert not p2.get_tag("capability:missing_values")
    p2.fit(X, y)

    # any regressor does not handle missing values
    p3 = RegressorEnsemble(regressors=[r1, r2, r1])
    assert not p3.get_tag("capability:missing_values")
    p3.fit(X, y)


def test_multivariate_tag_inference():
    """Test that RegressorEnsemble infers multivariate tag correctly."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=12, regression_target=True
    )

    r1 = MockRegressorFullTags()
    r2 = MockRegressor()

    assert r1.get_tag("capability:multivariate")
    assert not r2.get_tag("capability:multivariate")

    # regressors handle multivariate
    p1 = RegressorEnsemble(regressors=[r1, r1, r1])
    assert p1.get_tag("capability:multivariate")
    p1.fit(X, y)

    # test they fit even if they cannot handle multivariate
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)

    # regressors do not handle multivariate
    p2 = RegressorEnsemble(regressors=[r2, r2, r2])
    assert not p2.get_tag("capability:multivariate")
    p2.fit(X, y)

    # any regressor does not handle multivariate
    p3 = RegressorEnsemble(regressors=[r1, r2, r1])
    assert not p3.get_tag("capability:multivariate")
    p3.fit(X, y)
