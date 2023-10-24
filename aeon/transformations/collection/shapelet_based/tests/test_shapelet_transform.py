"""Test shapelet transform."""

import pytest

from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from aeon.utils._testing.collection import make_3d_test_data


def test_shapelet_transform():
    """
    Test edge cases for RandomShapeletTransform.

    This function tests the functionality of the RandomShapeletTransform class 
    from the aeon.transformations.collection.shapelet_based module. It uses the 
    make_3d_test_data function from the aeon.utils._testing.collection module 
    to generate test data. The function checks that the RandomShapeletTransform 
    class correctly handles edge cases, such as when the shortest series in 
    transform is smaller than the min shapelet length.
    """
    X, y = make_3d_test_data(n_cases=10, n_timepoints=20, n_labels=4)
    rst = RandomShapeletTransform(max_shapelets=3, remove_self_similar=True)
    rst._fit(X, y)
    # Assert at least one shapelet per class when max_shapelets < n_labels
    assert len(rst.shapelets) == 4
    X, y = make_3d_test_data(
        n_cases=3, n_timepoints=rst.max_fitted_shapelet_length_ - 1, n_labels=4
    )
    with pytest.raises(
        ValueError,
        match="The shortest series in transform is "
        "smaller than the min shapelet length",
    ):
        rst._transform(X)
