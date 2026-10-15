"""Test base segmenter."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from aeon.segmentation.base import BaseSegmenter
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)
from aeon.testing.mock_estimators import (
    MockSegmenter,
    MockSegmenterRequiresY,
    MockSegmenterWithScores,
)

N_TIMEPOINTS = 20
N_CHANNELS = 3
X_UNIVARIATE = make_example_1d_numpy(n_timepoints=N_TIMEPOINTS, random_state=0)
X_MULTIVARIATE = make_example_2d_numpy_series(
    n_timepoints=N_TIMEPOINTS, n_channels=N_CHANNELS, random_state=0
)


def test_fit_predict_correct():
    """
    Test returns self.

    Test same on two calls same input type.
    """
    x_correct = np.array([0, 0, 0, 1, 1])
    seg = MockSegmenter()
    res = seg.fit(x_correct)
    assert isinstance(res, BaseSegmenter)
    assert res.is_fitted
    assert res.axis == 1
    seg.set_tags(**{"fit_is_empty": True})
    res = seg.fit(x_correct)
    assert res.is_fitted
    res = seg.fit_predict(x_correct)
    assert isinstance(res, np.ndarray)
    seg = MockSegmenterRequiresY()
    res = seg.fit(x_correct, y=x_correct)
    assert res.is_fitted
    with pytest.raises(
        ValueError, match="Tag requires_y is true, but fit called with y=None"
    ):
        seg.fit(x_correct)


def test_to_classification():
    """Test class method to_classification."""
    labels = BaseSegmenter.to_classification([2, 8], length=10)
    assert isinstance(labels, np.ndarray)
    labels = BaseSegmenter.to_classification([2, 8], 10)
    assert np.array_equal(labels, np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0]))
    labels = BaseSegmenter.to_classification([0], 4)
    assert np.array_equal(labels, np.array([1, 0, 0, 0]))
    labels = BaseSegmenter.to_classification([0, 1, 2, 3], 4)
    assert np.array_equal(labels, np.array([1, 1, 1, 1]))


def test_to_clusters():
    """Test class method to_clusters."""
    labels = BaseSegmenter.to_clusters([2, 8], length=10)
    assert isinstance(labels, np.ndarray)
    labels = BaseSegmenter.to_clusters([2, 8], 10)
    assert np.array_equal(labels, np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2]))
    labels = BaseSegmenter.to_clusters([1, 2, 3], 4)
    assert np.array_equal(labels, np.array([0, 1, 2, 3]))


INPUT_CORRECT = [
    np.array([0, 0, 0, 1, 1]),
    pd.Series([0, 0, 0, 1, 1]),
    pd.DataFrame([0, 0, 0, 1, 1]),
]
INPUT_WRONG = [
    np.array([[0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2]]),
    pd.DataFrame([[0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2]]),
    np.array([0, 0, 0, 1, "FOO"]),
    pd.Series([0, 0, 0, 1, "FOO"]),
    pd.DataFrame([0, 0, 0, 1, "FOO"]),
    "Up the arsenal",
]


@pytest.mark.parametrize("y_correct", INPUT_CORRECT)
def test_check_y_correct(y_correct):
    """Test the _check_y method with correct input."""
    assert BaseSegmenter._check_y(None, y_correct) is None


@pytest.mark.parametrize("y_wrong", INPUT_WRONG)
def test_check_y_wrong(y_wrong):
    """Test the _check_y method with incorrect input."""
    with pytest.raises(ValueError, match="Error in input type for y"):
        BaseSegmenter._check_y(None, y_wrong)


@pytest.mark.parametrize("X", [X_UNIVARIATE, X_MULTIVARIATE])
def test_predict_scores_returns_one_score_per_timepoint(X):
    """Test predict_scores returns a float score for each time point of X."""
    seg = MockSegmenterWithScores().fit(X)
    scores = seg.predict_scores(X)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (N_TIMEPOINTS,)
    assert np.issubdtype(scores.dtype, np.floating)


def test_predict_scores_axis():
    """Test predict_scores gives the same scores for X with axis=1 and X.T, axis=0."""
    seg = MockSegmenterWithScores().fit(X_MULTIVARIATE)
    np.testing.assert_array_equal(
        seg.predict_scores(X_MULTIVARIATE, axis=1),
        seg.predict_scores(X_MULTIVARIATE.T, axis=0),
    )


def test_predict_scores_not_fitted():
    """Test predict_scores raises NotFittedError before fit if fit is not empty."""
    seg = MockSegmenterWithScores()
    with pytest.raises(NotFittedError):
        seg.predict_scores(X_UNIVARIATE)


def test_predict_scores_tag_false():
    """Test predict_scores raises NotImplementedError if the tag is False.

    The segmenter implements _predict_scores, so the error can only come from the
    tag check.
    """
    seg = MockSegmenterWithScores().fit(X_UNIVARIATE)
    seg.set_tags(**{"capability:predict_scores": False})
    with pytest.raises(NotImplementedError):
        seg.predict_scores(X_UNIVARIATE)


def test_predict_scores_tag_true_without_override():
    """Test predict_scores raises NotImplementedError if _predict_scores is missing.

    The tag is set to True on a segmenter that does not override _predict_scores, so
    the call reaches the base class _predict_scores.
    """
    seg = MockSegmenter().fit(X_UNIVARIATE)
    seg.set_tags(**{"capability:predict_scores": True})
    # Guard the premise: the call must reach the base class _predict_scores.
    assert "_predict_scores" not in type(seg).__dict__
    with pytest.raises(NotImplementedError):
        seg.predict_scores(X_UNIVARIATE)
