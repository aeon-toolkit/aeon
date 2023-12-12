"""Test base segmenter."""
import numpy as np
import pandas as pd
import pytest

from aeon.segmentation import BaseSegmenter


class TestSegmenter(BaseSegmenter):
    def _fit(self, X, y):
        pass

    def _predict(self, X):
        pass


def test_check_input():
    """Test check_input method.

    Only allow np.ndarray, pd.Series or pd.DataFrame of floats"""
    segmenter = TestSegmenter()
    assert segmenter._check_input_series(np.array([1, 2, 3])) is None
    with pytest.raises(ValueError, match="Error in input type"):
        segmenter._check_input_series(1)
    with pytest.raises(ValueError, match="Error in input type"):
        segmenter._check_input_series("a")
    with pytest.raises(ValueError, match="array must contain floats or ints"):
        segmenter._check_input_series(np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match="Should be 1D or 2D"):
        segmenter._check_input_series(np.random.random((5, 2, 5)))
    assert segmenter._check_input_series(pd.Series([1, 2, 3])) is None
    with pytest.raises(ValueError, match="pd.Series must be numeric"):
        segmenter._check_input_series(pd.Series(["a", 1.0, 1]))
    assert segmenter._check_input_series(pd.DataFrame([1, 2, 3])) is None
    with pytest.raises(ValueError, match="pd.DataFrame must be numeric"):
        segmenter._check_input_series(pd.DataFrame(["a", "b", 1.0], [1, 1, 1.0]))


def test_dummy():
    """Test dummy segmenter."""
    data = np.random.random((5, 100))  # 5 series of length 100
    segmenter = TestSegmenter()
    segmenter.fit(data)
    segs = segmenter.predict(data)
    assert len(segs) == 1
    segmenter = TestSegmenter(random_state=49, n_segments=10)
    segmenter.fit(data)
    assert segmenter.n_segments_ == 10
    segs = segmenter.predict(data)
    df = pd.DataFrame(data)
    segmenter = TestSegmenter(random_state=49, n_segments=10)
    segmenter.fit(df)
    # segs2 = segmenter.predict(df)
    # assert segs == segs2
