"""Test the setar-tree forecaster."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from aeon.forecasting._setar import SetartreeForecaster


def test_create_input_matrix_too_short():
    """Ensure that too-short series yields (None, None)."""
    # lag=3 but only 3 points => cannot create any samples
    model = SetartreeForecaster(lag=3)
    X, y = model._create_input_matrix(np.array([[1, 2, 3]]))
    assert X is None and y is None


def test_create_input_matrix_correct_shapes():
    """Verify correct shapes and flipped ordering for a valid input matrix."""
    # use lag=2 on a single series of length 5 => 5-2=3 samples of length 2
    model = SetartreeForecaster(lag=2)
    y = np.array([[10, 20, 30, 40, 50]])
    X, y_out = model._create_input_matrix(y)
    assert X.shape == (3, 2)
    assert y_out.shape == (3,)
    # first row of X should be [20,10] (flipped)
    np.testing.assert_array_equal(X[0], np.array([20, 10]))
    assert y_out[0] == 30


def test_fit_pr_model_and_calculate_sse():
    """Check that fitting a linear model and computing SSE works on y=2x."""
    model = SetartreeForecaster()
    # create a perfect linear relationship y=2x
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    lr = model._fit_pr_model(X, y)
    # Check that coefficient is close to 2
    assert pytest.approx(lr.coef_[0], rel=1e-3) == 2.0
    # perfect fit => zero SSE
    sse = model._calculate_sse(lr, X, y)
    assert pytest.approx(sse, abs=1e-6) == 0.0


def test_calculate_sse_empty_y():
    """Ensure SSE returns 0 when y is empty."""
    model = SetartreeForecaster()
    # should return 0 rather than crash
    lr = LinearRegression()
    sse = model._calculate_sse(lr, np.empty((0, 1)), np.array([]))
    assert sse == 0


def test_find_optimal_split_simple():
    """Confirm optimal split finds zero SSE on y=x for a single feature."""
    model = SetartreeForecaster()
    # single feature, y=x so best split at threshold=3 or 2 gives zero SSE
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])
    best = model._find_optimal_split(X, y)
    assert best["sse"] == pytest.approx(0.0)
    assert best["lag_idx"] == 0
    # threshold must be one of the unique values that yields two non-empty splits
    assert best["threshold"] in {2, 3}


def test_check_linearity_degrees_of_freedom():
    """Check that linearity test returns False when df_child ≤ 0."""
    # for df_child <= 0 should immediately return False
    model = SetartreeForecaster(lag=5)
    parent_sse = 100
    child_sse = 80
    n_samples = 10  # df_child = 10 - 2*5 -2 = -2
    assert (
        model._check_linearity(parent_sse, child_sse, n_samples, current_alpha=0.05)
        is False
    )


def test_check_error_improvement():
    """Verify error improvement logic against the threshold."""
    model = SetartreeForecaster(error_threshold=0.10)
    # improvement = (100-85)/100 = 0.15 >= 0.10
    assert model._check_error_improvement(100, 85) is True
    # improvement = (100-95)/100 = 0.05 < 0.10
    assert model._check_error_improvement(100, 95) is False


def test_fit_short_series_creates_single_leaf():
    """Ensure a tiny series still yields at least one leaf model."""
    # With lag=2 and series length=3, embedded matrix exists,
    # but splitting may not happen
    model = SetartreeForecaster(lag=2, max_depth=1, stopping_criteria="both")
    y = np.array([[1, 2, 3]])
    fitted = model._fit(y)
    # tree_ must be a dict and should mark root as leaf if no split happened
    assert isinstance(fitted.tree_, dict)
    # either the root is a leaf or there is at least one leaf model
    assert fitted.leaf_models_


def test_predict_in_sample_basic():
    """Verify in-sample prediction returns the next true value for minimal data."""
    # use lag=3 so that a series of length 4 yields exactly one training sample
    model = SetartreeForecaster(lag=3, horizon=1)
    # series = [1,2,3,4] → X=[[1,2,3]] → y=[4]
    y = np.array([[1, 2, 3, 4]])
    model.fit(y)

    pred = model._predict()  # in‐sample path
    # should be a numpy scalar/0‐d array equal to the true next value 4
    assert isinstance(pred, np.ndarray)
    assert pred.shape == ()  # scalar array
    assert pred.item() == 4
