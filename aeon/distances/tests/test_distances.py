"""Tests for computing distances."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import alignment_path, cost_matrix
from aeon.distances import distance as compute_distance
from aeon.distances import get_distance_function_names, pairwise_distance
from aeon.distances._distance import (
    DISTANCES,
    MIN_DISTANCES,
    MP_DISTANCES,
    SINGLE_POINT_NOT_SUPPORTED_DISTANCES,
    UNEQUAL_LENGTH_SUPPORT_DISTANCES,
    _custom_func_pairwise,
    _resolve_key_from_distance,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)
from aeon.testing.expected_results.expected_distance_results import (
    _expected_distance_results,
)

UNEQUAL_LENGTH_NOT_SUPPORTED_DISTANCES = ["shift_scale"]


def _validate_distance_result(
    x, y, name, distance, symmetric, expected_result=10, check_xy_permuted=True
):
    """
    Validate the distance result by comparing it with the expected result.

    Parameters
    ----------
    x (np.ndarray): First array.
    y (np.ndarray): Second array.
    name (str): Name of the distance method.
    distance (callable): Distance function.
    expected_result (float): Expected distance result.
    check_xy_permuted: (bool): recursively call with swapped series
    """
    original_x = x.copy()
    original_y = y.copy()
    if expected_result is None:
        return

    dist_result = distance(x, y)
    assert isinstance(dist_result, float)
    assert_almost_equal(dist_result, expected_result)
    assert_almost_equal(dist_result, compute_distance(x, y, method=name))
    assert_almost_equal(dist_result, compute_distance(x, y, method=distance))

    dist_result_to_self = distance(x, x)
    assert isinstance(dist_result_to_self, float)

    # If unequal length swap where x and y are to ensure it works both ways around

    if symmetric and original_x.shape[-1] != original_y.shape[-1] and check_xy_permuted:
        _validate_distance_result(
            original_y,
            original_x,
            name,
            distance,
            symmetric,
            expected_result,
            check_xy_permuted=False,
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_distances(dist):
    """Test distance functions."""
    # For now skipping mpdist and mindist
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return

    # ================== Test equal length ==================
    # Test univariate of shape (n_timepoints,)
    _validate_distance_result(
        make_example_1d_numpy(10, random_state=1),
        make_example_1d_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["symmetric"],
        _expected_distance_results[dist["name"]][0],
    )

    # Test univariate of shape (1, n_timepoints)
    _validate_distance_result(
        make_example_2d_numpy_series(10, 1, random_state=1),
        make_example_2d_numpy_series(10, 1, random_state=2),
        dist["name"],
        dist["distance"],
        dist["symmetric"],
        _expected_distance_results[dist["name"]][0],
    )

    # Test multivariate of shape (n_channels, n_timepoints)
    _validate_distance_result(
        make_example_2d_numpy_series(10, 1, random_state=1),
        make_example_2d_numpy_series(10, 1, random_state=2),
        dist["name"],
        dist["distance"],
        dist["symmetric"],
        _expected_distance_results[dist["name"]][1],
    )

    # ================== Test unequal length ==================
    if dist["name"] in UNEQUAL_LENGTH_SUPPORT_DISTANCES:
        # Test univariate unequal length of shape (n_timepoints,)
        _validate_distance_result(
            make_example_1d_numpy(5, random_state=1),
            make_example_1d_numpy(10, random_state=2),
            dist["name"],
            dist["distance"],
            dist["symmetric"],
            _expected_distance_results[dist["name"]][2],
        )

        # Test univariate unequal length of shape (1, n_timepoints)
        _validate_distance_result(
            make_example_2d_numpy_series(5, 1, random_state=1),
            make_example_2d_numpy_series(10, 1, random_state=2),
            dist["name"],
            dist["distance"],
            dist["symmetric"],
            _expected_distance_results[dist["name"]][2],
        )

        # Test multivariate unequal length of shape (n_channels, n_timepoints)
        _validate_distance_result(
            make_example_2d_numpy_series(5, 5, random_state=1),
            make_example_2d_numpy_series(10, 10, random_state=2),
            dist["name"],
            dist["distance"],
            dist["symmetric"],
            _expected_distance_results[dist["name"]][3],
        )

    # ============== Test single point series ==============
    if dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES:
        # Test singe point univariate of shape (1,)
        _validate_distance_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["symmetric"],
            _expected_distance_results[dist["name"]][4],
        )

        # Test singe point univariate of shape (1, 1)
        _validate_distance_result(
            np.array([[10.0]]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["symmetric"],
            _expected_distance_results[dist["name"]][4],
        )


def test_get_distance_function_names():
    """Test get_distance_function_names."""
    assert get_distance_function_names() == sorted([dist["name"] for dist in DISTANCES])


def test_resolve_key_from_distance():
    """Test _resolve_key_from_distance."""
    with pytest.raises(ValueError, match="Unknown method"):
        _resolve_key_from_distance(method="FOO", key="cost_matrix")
    with pytest.raises(ValueError):
        _resolve_key_from_distance(method="dtw", key="FOO")

    def foo(x, y):
        return 0

    assert callable(_resolve_key_from_distance(foo, key="FOO"))


def test_incorrect_inputs():
    """Test the handling of incorrect inputs."""
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    with pytest.raises(
        ValueError,
        match="Method must be one of the supported strings or a " "callable",
    ):
        compute_distance(x, y, method="FOO")
    with pytest.raises(
        ValueError,
        match="Method must be one of the supported strings or a " "callable",
    ):
        pairwise_distance(x, y, method="FOO")
    with pytest.raises(ValueError, match="Method must be one of the supported strings"):
        alignment_path(x, y, method="FOO")
    with pytest.raises(ValueError, match="Method must be one of the supported strings"):
        cost_matrix(x, y, method="FOO")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError, match="dist_func must be a callable"):
        _custom_func_pairwise(x, dist_func=None)
