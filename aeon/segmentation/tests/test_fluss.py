"""Simple FLUSS test."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.datasets import load_gun_point_segmentation
from aeon.segmentation import FLUSSSegmenter
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["stumpy"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fluss_sparse():
    """Test FLUSS segmentation.

    Check if the predicted change points match.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a FLUSS segmentation
    fluss = FLUSSSegmenter(period_size, n_regimes=2)
    found_cps = fluss.fit_predict(ts)
    scores = fluss.predict_scores(ts)

    assert len(found_cps) == 1 and found_cps[0] == 889
    assert len(scores) == 1 and 0.53 > scores[0] > 0.52
