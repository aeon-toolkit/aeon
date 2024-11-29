"""Test Warping series transformer."""

__maintainer__ = ["hadifawaz1999"]

import pytest

from aeon.clustering.averaging import VALID_BA_METRICS
from aeon.distances import get_alignment_path_function
from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.transformations.series import WarpingSeriesTransformer


@pytest.mark.parametrize("distance", VALID_BA_METRICS)
def test_warping_path_transformer(distance):
    """Test the functionality of Warping transformation."""
    x = make_example_2d_numpy_series(n_timepoints=20, n_channels=2)
    y = make_example_2d_numpy_series(n_timepoints=20, n_channels=2)

    alignment_path_function = get_alignment_path_function(method=distance)

    warping_path = alignment_path_function(x, y)[0]

    new_x = WarpingSeriesTransformer(
        series_index=0, warping_path=warping_path
    ).fit_transform(x)
    new_y = WarpingSeriesTransformer(
        series_index=1, warping_path=warping_path
    ).fit_transform(y)

    assert int(new_x.shape[1]) == len(warping_path)
    assert int(new_y.shape[1]) == len(warping_path)
