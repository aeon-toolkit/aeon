"""Simple BinSeg test."""

__maintainer__ = []
__all__ = []

from aeon.datasets import load_gun_point_segmentation
from aeon.segmentation import BinSegSegmenter


def test_binseg_sparse():
    """Test BinSeg sparse segmentation.

    Check if the predicted change points match.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a BinSeg segmentation
    binseg = BinSegSegmenter(period_size, n_cps=1)
    found_cps = binseg.fit_predict(ts)

    assert len(found_cps) == 1 and found_cps[0] == 1750
