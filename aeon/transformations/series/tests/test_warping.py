"""Test Warping series transformer."""

__maintainer__ = ["hadifawaz1999"]

import numpy as np
import pytest

from aeon.clustering.averaging import VALID_BA_DISTANCE_METHODS
from aeon.distances import get_alignment_path_function
from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.transformations.series import WarpingSeriesTransformer


@pytest.mark.parametrize("distance", VALID_BA_DISTANCE_METHODS)
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


def test_warping_path_none_returns_input_unchanged():
    """Test warping is an identity transform when no path is supplied."""
    n_timepoints = 20
    n_channels = 2
    x = make_example_2d_numpy_series(n_timepoints=n_timepoints, n_channels=n_channels)
    new_x = WarpingSeriesTransformer(warping_path=None).fit_transform(x)
    np.testing.assert_array_equal(new_x, x)


def test_warping_invalid_series_index_raises():
    """Test warping rejects ``series_index`` values outside {0, 1}."""
    n_timepoints = 20
    n_channels = 2
    x = make_example_2d_numpy_series(n_timepoints=n_timepoints, n_channels=n_channels)
    y = make_example_2d_numpy_series(n_timepoints=n_timepoints, n_channels=n_channels)
    warping_path = get_alignment_path_function(method="dtw")(x, y)[0]

    transformer = WarpingSeriesTransformer(series_index=2, warping_path=warping_path)
    with pytest.raises(ValueError, match="0 or 1"):
        transformer.fit_transform(x)
