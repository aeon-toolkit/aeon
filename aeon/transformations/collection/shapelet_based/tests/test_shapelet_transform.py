"""Tests for RandomShapeletTransform."""

__maintainer__ = ["MatthewMiddlehurst"]

import pytest

from aeon.datasets import load_unit_test
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform


def test_shapelet_transform_default_quality_measure():
    """Test that default quality measure is information_gain."""
    t = RandomShapeletTransform(n_shapelet_samples=10, max_shapelets=5)
    assert t.quality_measure == "information_gain"


def test_shapelet_transform_quality_measure_validation():
    """Test that invalid quality measures raise ValueError."""
    with pytest.raises(ValueError, match="quality_measure must be one of"):
        RandomShapeletTransform(quality_measure="invalid_measure")


def test_shapelet_transform_information_gain():
    """Test RandomShapeletTransform with information_gain quality measure."""
    X_train, y_train = load_unit_test(split="train")

    t = RandomShapeletTransform(
        n_shapelet_samples=20,
        max_shapelets=5,
        quality_measure="information_gain",
        random_state=42,
    )
    t.fit(X_train, y_train)
    X_t = t.transform(X_train)

    assert X_t.shape[0] == len(X_train)
    assert X_t.shape[1] == len(t.shapelets)
    assert len(t.shapelets) > 0


def test_shapelet_transform_f_statistic():
    """Test RandomShapeletTransform with f_statistic quality measure."""
    X_train, y_train = load_unit_test(split="train")

    t = RandomShapeletTransform(
        n_shapelet_samples=20,
        max_shapelets=5,
        quality_measure="f_statistic",
        random_state=42,
    )
    t.fit(X_train, y_train)
    X_t = t.transform(X_train)

    assert X_t.shape[0] == len(X_train)
    assert X_t.shape[1] == len(t.shapelets)
    assert len(t.shapelets) > 0


def test_shapelet_transform_backward_compatibility():
    """Test that default behavior matches explicit information_gain."""
    X_train, y_train = load_unit_test(split="train")

    # Default (should use information_gain)
    t1 = RandomShapeletTransform(
        n_shapelet_samples=20, max_shapelets=5, random_state=42
    )
    t1.fit(X_train, y_train)
    X_t1 = t1.transform(X_train)

    # Explicit information_gain
    t2 = RandomShapeletTransform(
        n_shapelet_samples=20,
        max_shapelets=5,
        quality_measure="information_gain",
        random_state=42,
    )
    t2.fit(X_train, y_train)
    X_t2 = t2.transform(X_train)

    # Should produce same results with same random_state
    assert X_t1.shape == X_t2.shape
    assert len(t1.shapelets) == len(t2.shapelets)


def test_shapelet_transform_different_quality_measures():
    """Test that different quality measures produce valid outputs.

    Both information_gain and f_statistic should produce valid but
    potentially different results.
    """
    X_train, y_train = load_unit_test(split="train")

    # Information gain
    t1 = RandomShapeletTransform(
        n_shapelet_samples=20,
        max_shapelets=5,
        quality_measure="information_gain",
        random_state=42,
    )
    t1.fit(X_train, y_train)
    X_t1 = t1.transform(X_train)

    # F-statistic
    t2 = RandomShapeletTransform(
        n_shapelet_samples=20,
        max_shapelets=5,
        quality_measure="f_statistic",
        random_state=42,
    )
    t2.fit(X_train, y_train)
    X_t2 = t2.transform(X_train)

    # Both should produce valid outputs
    assert X_t1.shape[0] == X_t2.shape[0] == len(X_train)
    assert len(t1.shapelets) > 0
    assert len(t2.shapelets) > 0
    # Note: Results may differ due to different quality measures
