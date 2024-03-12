import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import alignment_path as compute_alignment_path
from aeon.distances._distance import DISTANCES
from aeon.testing.utils.data_gen import make_series


def _validate_cost_matrix_result(
    x: np.ndarray,
    y: np.ndarray,
    name,
    distance,
    alignment_path,
):
    alignment_path_result = alignment_path(x, y)

    assert isinstance(alignment_path_result, tuple)
    assert isinstance(alignment_path_result[0], list)
    assert isinstance(alignment_path_result[1], float)
    assert compute_alignment_path(x, y, metric=name) == alignment_path_result

    distance_result = distance(x, y)
    assert_almost_equal(alignment_path_result[1], distance_result)


COST_MATRIX_SINGLE_NOT_SUPPORTED = ["ddtw", "wddtw", "lcss"]


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    if "alignment_path" not in dist:
        return

    # ================== Test equal length ==================
    # Test univariate of shape (n_timepoints,)
    _validate_cost_matrix_result(
        make_series(10, return_numpy=True, random_state=1),
        make_series(10, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test univariate of shape (1, n_timepoints)
    _validate_cost_matrix_result(
        make_series(10, 1, return_numpy=True, random_state=1),
        make_series(10, 1, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test multivariate unequal length of shape (n_channels, n_timepoints)
    _validate_cost_matrix_result(
        make_series(10, 10, return_numpy=True, random_state=1),
        make_series(10, 10, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # ================== Test unequal length ==================
    # Test univariate unequal length of shape (n_timepoints,)
    _validate_cost_matrix_result(
        make_series(5, return_numpy=True, random_state=1),
        make_series(10, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test univariate unequal length of shape (1, n_timepoints)
    _validate_cost_matrix_result(
        make_series(5, 1, return_numpy=True, random_state=1),
        make_series(10, 1, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test multivariate unequal length of shape (n_channels, n_timepoints)
    _validate_cost_matrix_result(
        make_series(5, 10, return_numpy=True, random_state=1),
        make_series(10, 10, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # ============== Test single point series ==============
    if dist["name"] not in COST_MATRIX_SINGLE_NOT_SUPPORTED:
        # Test singe point univariate of shape (n_timepoints,)
        _validate_cost_matrix_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )

        # Test singe point univariate of shape (1, n_timepoints)
        _validate_cost_matrix_result(
            np.array([[10.0]]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )
