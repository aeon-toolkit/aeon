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


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch is not available",
)
def test_shapeleter_classifier_fit_predict():
    """Test that ShapeleterClassifier fits and predicts correctly."""
    import numpy as np

    X = np.random.RandomState(42).normal(size=(10, 1, 30))
    y = np.array([0, 1] * 5)
    clf = ShapeleterClassifier(max_shapelets=20, max_iter=5, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == 10
    assert set(y_pred).issubset({0, 1})
