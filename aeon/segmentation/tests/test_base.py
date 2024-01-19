"""Test base segmenter."""
import numpy as np
import pandas as pd
import pytest

from aeon.segmentation import BaseSegmenter
from aeon.testing.mock_estimators import MockSegmenter, SupervisedMockSegmenter

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
def test__check_y_correct(y_correct):
    seg = SupervisedMockSegmenter()
    assert seg._check_y(y_correct) is None


@pytest.mark.parametrize("y_wrong", INPUT_WRONG)
def test__check_y_wrong(y_wrong):
    seg = SupervisedMockSegmenter()
    with pytest.raises(ValueError, match="Error in input type for y"):
        seg._check_y(y_wrong)


def test_fit_predict_correct():
    """Test returns self.
    Test same on two calls same input type

    """
    x_correct = np.array([0, 0, 0, 1, 1])
    seg = MockSegmenter()
    res = seg.fit(x_correct)
    assert isinstance(res, BaseSegmenter)
    assert res.is_fitted
    assert res.axis == 0
    seg.set_tags(**{"fit_is_empty": True})
    res = seg.fit(x_correct)
    assert res.is_fitted
    res = seg.fit_predict(x_correct)
    assert isinstance(res, np.ndarray)
    seg = SupervisedMockSegmenter()
    res = seg.fit(x_correct, y=x_correct)
    assert res.is_fitted
    with pytest.raises(
        ValueError, match="Tag requires_y is true, but fit called with y=None"
    ):
        seg.fit(x_correct)


def test_to_classification():
    pass
    # labels = BaseSegmenter.to_classification(None, [2, 8])
    # assert labels not None


#    assert labels == np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])


def test_to_clusters():
    pass
    # labels = BaseSegmenter.to_classification(None, [2, 8])
    # assert labels not None
