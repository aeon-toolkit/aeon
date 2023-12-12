"""Test base segmenter."""
import numpy as np
import pandas as pd
import pytest

from aeon.segmentation import BaseSegmenter, DummySegmenter


class TestSegmenter(BaseSegmenter):
    def _fit(self, X, y):
        pass

    def _predict(self, X):
        pass


class TestMultivariateSegmenter(BaseSegmenter):
    _tags = {
        "capability:multivariate": True,
    }

    def _fit(self, X, y):
        pass

    def _predict(self, X):
        pass


def test__check_input():
    """Test check_input method."""

    # Allow np.ndarray, pd.Series or pd.DataFrame of floats
    segmenter = TestSegmenter()
    assert segmenter._check_input_series(np.array([1, 2, 3])) is None
    assert segmenter._check_input_series(pd.Series([1, 2, 3])) is None
    assert segmenter._check_input_series(pd.DataFrame([1, 2, 3])) is None

    # Dont allow other types
    with pytest.raises(ValueError, match="Error in input type"):
        segmenter._check_input_series(1)
    with pytest.raises(ValueError, match="Error in input type"):
        segmenter._check_input_series("a")
    with pytest.raises(ValueError, match="array must contain floats or ints"):
        segmenter._check_input_series(np.array(["a", "b", "c"]))

    # Internal format is correct
    with pytest.raises(ValueError, match="Should be 1D or 2D"):
        segmenter._check_input_series(np.random.random((5, 2, 5)))
    with pytest.raises(ValueError, match="pd.Series must be numeric"):
        segmenter._check_input_series(pd.Series(["a", 1.0, 1]))
    with pytest.raises(ValueError, match="pd.DataFrame must be numeric"):
        segmenter._check_input_series(pd.DataFrame(["a", "b", 1.0], [1, 1, 1.0]))


testy = np.random.random(10)
uni = []
uni.append(testy)
uni.append(pd.Series(testy))
uni.append(pd.DataFrame(testy))
# uni.append(np.random.random((10, 1)))
testy = np.random.random((4, 10))
multi = []
multi.append(testy)
multi.append(pd.DataFrame(testy))


def test__check_capabilities():
    """Check it deals with multivariate if its allowed."""
    seg1 = TestSegmenter()
    seg2 = TestMultivariateSegmenter()
    """ All should allow univariate ndarray, Series or single column DataFrame."""
    for u in uni:
        seg1._check_capabilities(u, axis=0)
        seg2._check_capabilities(u, axis=0)
    with pytest.raises(ValueError, match="Multivariate data not supported"):
        seg1._check_capabilities(pd.DataFrame(testy), axis=1)
    for m in multi:
        seg2._check_capabilities(m, axis=0)
        seg2._check_capabilities(m, axis=1)
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            seg1._check_capabilities(m, axis=0)
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            seg1._check_capabilities(m, axis=1)


def test__convert_X():
    """Test _convert_X method."""
    segmenter = TestSegmenter()
    segmenter._tags["X_inner_type"] = "ndarray"
    segmenter.axis = 0
    for axis in [0, 1]:
        for u in uni:
            res = segmenter._convert_X(u, axis)
            assert isinstance(res, np.ndarray)
            assert len(res) == 10
            assert len(res.shape) == 1
    for m in multi:
        res = segmenter._convert_X(m, 0)
        assert isinstance(res, np.ndarray)
        assert res.shape == (4, 10)
    for m in multi:
        res = segmenter._convert_X(m, 1)
        assert isinstance(res, np.ndarray)
        assert res.shape == (10, 4)
    segmenter._tags["X_inner_type"] = "Series"
    for axis in [0, 1]:
        for u in uni:
            res = segmenter._convert_X(u, axis)
            assert isinstance(res, pd.Series)
            assert len(res) == 10


def test_dummy():
    """Test dummy segmenter."""
    data = np.random.random((5, 100))  # 5 series of length 100
    segmenter = DummySegmenter()
    segmenter.fit(data)
    segs = segmenter.predict(data)
    assert len(segs) == 1
    segmenter = DummySegmenter(random_state=49, n_segments=10)
    segmenter.fit(data)
    assert segmenter.n_segments_ == 10
    segs = segmenter.predict(data)
    df = pd.DataFrame(data)
    segmenter = DummySegmenter(random_state=49, n_segments=10)
    segmenter.fit(df)
    # segs2 = segmenter.predict(df)
    # assert segs == segs2
