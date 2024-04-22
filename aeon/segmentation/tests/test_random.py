"""Tests for random segmenter."""

import numpy as np
import pandas as pd

from aeon.segmentation import RandomSegmenter


def test_random():
    """Test random segmenter."""
    data = np.random.random((5, 100))  # 5 series of length 100
    segmenter = RandomSegmenter()
    segmenter.fit(data)
    segs = segmenter.predict(data)
    assert len(segs) == 1
    segmenter = RandomSegmenter(random_state=49, n_segments=10)
    segmenter.fit(data)
    assert segmenter.n_segments == 10
    segs2 = segmenter.predict(data)
    np.array_equal(segs, segs2)
    assert len(segs) == 1
    segs = segmenter.fit_predict(data)
    assert len(segs) == 9

    df = pd.DataFrame(data)
    segmenter = RandomSegmenter(random_state=49, n_segments=10)
    segmenter.fit(df)
    segs2 = segmenter.predict(df)
    np.array_equal(segs, segs2)
