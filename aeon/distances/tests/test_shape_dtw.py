"""Tests for computing shape dtw measure."""

from aeon.distances import shape_dtw_distance
from aeon.distances._shape_dtw import _pad_ts_edges, _transform_subsequences
from aeon.testing.utils.data_gen import make_example_3d_numpy


def test_precomputed_transformation_shapedtw():
    """Test precomputation of transformation."""
    X, _ = make_example_3d_numpy()

    shape_dtw_default = shape_dtw_distance(
        x=X[0],
        y=X[1],
        reach=4,
        transformation_precomputed=False,
    )

    padded_x = _pad_ts_edges(x=X[0], reach=4)
    padded_y = _pad_ts_edges(x=X[1], reach=4)

    transformed_x = _transform_subsequences(x=padded_x, reach=4)
    transformed_y = _transform_subsequences(x=padded_y, reach=4)

    shape_dtw_precomputed_transformation = shape_dtw_distance(
        x=X[0],
        y=X[1],
        reach=10,
        transformation_precomputed=True,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
    )

    assert shape_dtw_default == shape_dtw_precomputed_transformation
