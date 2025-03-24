"""Simple BinSeg test."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.datasets import load_gun_point_segmentation
from aeon.segmentation import BinSegmenter
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["ruptures"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_binseg_sparse():
    """Test BinSeg sparse segmentation.

    Check if the predicted change points match.
    """
    # load the test dataset
    ts, _, cps = load_gun_point_segmentation()

    # compute a BinSeg segmentation
    binseg = BinSegmenter(n_cps=1)
    found_cps = binseg.fit_predict(ts)

    assert len(found_cps) == 1 and found_cps[0] == 1870
