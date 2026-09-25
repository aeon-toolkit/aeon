"""Simple FLUSS test."""

__maintainer__ = []
__all__ = []

from copy import deepcopy

import pytest

from aeon.datasets import load_gun_point_segmentation
from aeon.segmentation import FLUSSSegmenter
from aeon.testing.utils.deep_equals import deep_equals
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


@pytest.mark.skipif(
    not _check_soft_dependencies(["stumpy"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fluss_predict_does_not_change_state():
    """Test predict and predict_scores do not store results on the estimator.

    Regression test for #3819: predict-type methods must not change the
    estimator's attributes.
    """
    ts, period_size, _ = load_gun_point_segmentation()
    fluss = FLUSSSegmenter(period_size, n_regimes=2).fit(ts)

    state_before = deepcopy(fluss.__dict__)
    found_cps = fluss.predict(ts)
    scores = fluss.predict_scores(ts)

    assert deep_equals(state_before, fluss.__dict__)
    for attr in ("found_cps", "profiles", "scores", "profile"):
        assert not hasattr(fluss, attr)
    # nothing is fitted, so there are no fitted parameters to return
    assert fluss.get_fitted_params() == {}
    # results are still returned correctly
    assert len(found_cps) == 1 and found_cps[0] == 889
    assert len(scores) == 1 and 0.53 > scores[0] > 0.52
