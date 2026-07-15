"""Tests for the BaseRotationForest class."""

import numpy as np
import pytest

from aeon.classification.sklearn import RotationForestClassifier
from aeon.regression.sklearn import RotationForestRegressor
from aeon.testing.data_generation import make_example_2d_numpy_collection

n_cases = 10
n_channels = 4
n_timepoints = 12


def _example_data(cls):
    return make_example_2d_numpy_collection(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
        regression_target=cls is RotationForestRegressor,
        random_state=0,
    )


@pytest.mark.parametrize("cls", [RotationForestClassifier, RotationForestRegressor])
def test_rotf_input(cls):
    """Test RotF rejects unsupported input shapes and degenerate data."""
    rotf = cls()

    # a univariate 3d array is squeezed to 2d
    X = rotf._check_X(np.random.random((n_cases, 1, n_timepoints)))
    assert X.shape == (n_cases, n_timepoints)

    # multivariate 3d and ragged inputs are rejected
    with pytest.raises(ValueError, match="not a time series"):
        rotf._check_X(np.random.random((n_cases, n_channels, n_timepoints)))
    with pytest.raises(ValueError, match="not a time series"):
        rotf._check_X([[1, 2, 3], [4, 5], [6, 7, 8]])

    # constant attributes leave nothing to fit on
    X_constant = np.zeros((n_cases, n_timepoints))
    y = np.zeros(n_cases)
    y[: n_cases // 2] = 1

    with pytest.raises(ValueError, match="same value"):
        rotf.fit_predict(X_constant, y)


def test_rotf_pca_solver_is_deprecated():
    """Test setting the deprecated pca_solver warns and has no effect on output."""
    X, y = _example_data(RotationForestClassifier)

    rotf_default = RotationForestClassifier(n_estimators=5, random_state=0)
    rotf_default.fit(X, y)

    rotf_solver = RotationForestClassifier(
        n_estimators=5,
        pca_solver="full",
        random_state=0,
    )
    with pytest.warns(DeprecationWarning, match="pca_solver"):
        rotf_solver.fit(X, y)

    np.testing.assert_array_equal(
        rotf_default.predict_proba(X),
        rotf_solver.predict_proba(X),
    )


@pytest.mark.parametrize(
    "cls,criterion,default_criterion",
    [
        (RotationForestClassifier, "gini", "entropy"),
        (RotationForestRegressor, "friedman_mse", "squared_error"),
    ],
)
def test_rotf_tree_parameters(cls, criterion, default_criterion):
    """Test exposed tree parameters reach the default decision trees."""
    X, y = _example_data(cls)

    n_estimators = 3
    max_depth = 3
    min_samples_leaf = 2

    rotf = cls(
        n_estimators=n_estimators,
        criterion=criterion,
        splitter="random",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    rotf.fit(X, y)

    assert len(rotf.estimators_) == n_estimators
    for tree in rotf.estimators_:
        assert tree.criterion == criterion
        assert tree.splitter == "random"
        assert tree.max_depth == max_depth
        assert tree.min_samples_leaf == min_samples_leaf

    default = cls(n_estimators=n_estimators, random_state=0)
    default.fit(X, y)
    assert default.estimators_[0].criterion == default_criterion
    assert default.estimators_[0].splitter == "best"
    assert default.estimators_[0].max_depth is None
