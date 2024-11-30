"""Simple AutoPlait test."""

import pytest

from aeon.datasets import load_gun_point_segmentation
from aeon.segmentation import AutoPlaitSegmenter


def test_autoplait_sparse():
    """Test AutoPlait segmentation.

    Check if the predicted change points match.
    """

    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a FLUSS segmentation
    autoplait = AutoPlaitSegmenter()
    #found_cps = autoplait.fit_predict(ts)
    return