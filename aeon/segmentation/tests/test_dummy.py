import numpy as np
import pandas as pd

from aeon.segmentation import DummySegmenter


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
