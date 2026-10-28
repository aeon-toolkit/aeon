"""Tests for Shapeleter classifier."""

import pytest

from aeon.classification.shapelet_based._shapeleter import (
    ShapeleterClassifier,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch is not available",
)
def test_shapeleter_classifier_parameter_routing():
    """Test that ShapeleterClassifier parameters route correctly."""
    clf = ShapeleterClassifier(
        max_shapelets=25,
        ka=6.0,
        ko=7.0,
        k_min=2.0,
        k_max=12.0,
        lr=0.005,
        max_iter=10,
        save_rate=0.4,
        random_state=123,
    )
    assert clf.max_shapelets == 25
    assert clf.ka == 6.0
    assert clf.ko == 7.0
    assert clf.k_min == 2.0
    assert clf.k_max == 12.0
    assert clf.lr == 0.005
    assert clf.max_iter == 10
    assert clf.save_rate == 0.4
    assert clf.random_state == 123


def test_shapeleter_classifier_torch_dependency():
    """Test that ShapeleterClassifier declares torch dependency in its tags."""
    assert ShapeleterClassifier.get_class_tag("python_dependencies") == "torch"
