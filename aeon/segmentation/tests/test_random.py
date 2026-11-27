"""Tests for random segmenter."""

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("n_timepoints", [2, 5, 10])
def test_random_nonempty_segments(n_timepoints):
    """Each timepoint can form a segment without selecting a cut at zero."""
    data = np.arange(n_timepoints, dtype=float)
    segmenter = RandomSegmenter(random_state=1, n_segments=n_timepoints)

    change_points = segmenter.fit_predict(data)

    np.testing.assert_array_equal(change_points, np.arange(1, n_timepoints))


def test_random_too_many_segments():
    """Reject requests that would require an empty segment."""
    segmenter = RandomSegmenter(random_state=1, n_segments=6)

    with pytest.raises(ValueError):
        segmenter.fit(np.arange(5, dtype=float))
