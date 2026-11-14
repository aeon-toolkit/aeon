"""Regression tests for segmentation estimator checks."""

import pytest

from aeon.segmentation import BaseSegmenter
from aeon.testing.estimator_checking._yield_segmentation_checks import (
    check_segmenter_base_functionality,
)


@pytest.mark.parametrize("fit_is_empty", [True, False])
@pytest.mark.parametrize("implements_fit", [True, False])
def test_segmenter_fit_is_empty_contract(fit_is_empty, implements_fit):
    """Reject inconsistent fit tags and accept both valid combinations."""

    def _predict(self, X):
        return [1]

    def _fit(self, X, y=None):
        return self

    attributes = {"_tags": {"fit_is_empty": fit_is_empty}, "_predict": _predict}
    if implements_fit:
        attributes["_fit"] = _fit
    segmenter_class = type("TestSegmenter", (BaseSegmenter,), attributes)

    if fit_is_empty == implements_fit:
        with pytest.raises(AssertionError, match="fit_is_empty"):
            check_segmenter_base_functionality(segmenter_class)
    else:
        check_segmenter_base_functionality(segmenter_class)
