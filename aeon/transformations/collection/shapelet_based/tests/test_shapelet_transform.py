"""Test shapelet transform."""

import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform


def test_shapelet_transform():
    """Test edge cases for RandomShapeletTransform."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=20, n_labels=4)
    rst = RandomShapeletTransform(max_shapelets=3, remove_self_similar=True)
    rst._fit(X, y)
    # Assert at least one shapelet per class when max_shapelets < n_labels
    assert len(rst.shapelets) == 4
    X, y = make_example_3d_numpy(
        n_cases=3, n_timepoints=rst.max_shapelet_length_ - 1, n_labels=4
    )
    with pytest.raises(
        ValueError,
        match="The shortest series in transform is "
        "smaller than the min shapelet length",
    ):
        rst._transform(X)
