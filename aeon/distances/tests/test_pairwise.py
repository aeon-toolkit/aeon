import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import pairwise_distance as compute_pairwise_distance
from aeon.distances._distance import DISTANCES
from aeon.distances.tests.test_utils import (
    SINGLE_POINT_NOT_SUPPORTED_DISTANCES,
    _make_3d_series,
)
from aeon.testing.utils.data_gen import (
    make_example_2d_numpy,
    make_example_3d_numpy,
    make_series,
)


def _validate_pairwise_result(
    x: np.ndarray,
    name,
    distance,
    pairwise_distance,
):
    pairwise_result = pairwise_distance(x)

    expected_size = (len(x), len(x))

    assert isinstance(pairwise_result, np.ndarray)
    assert pairwise_result.shape == expected_size
    assert_almost_equal(pairwise_result, compute_pairwise_distance(x, metric=name))
    assert_almost_equal(pairwise_result, compute_pairwise_distance(x, metric=distance))

    x = _make_3d_series(x)

    matrix = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(x)):
            curr_y = x[j]
            matrix[i, j] = distance(curr_x, curr_y)

    assert np.allclose(matrix, pairwise_result)


def _validate_multiple_to_multiple_result(
    x,
    y,
    name,
    distance,
    multiple_to_multiple_distance,
    recursive_call=False,
):
    original_x = x.copy()
    original_y = y.copy()
    multiple_to_multiple_result = multiple_to_multiple_distance(x, y)

    if x.ndim == 1 and y.ndim == 1:
        expected_size = (1, 1)
    else:
        expected_size = (len(x), len(y))

    assert isinstance(multiple_to_multiple_result, np.ndarray)
    assert multiple_to_multiple_result.shape == expected_size

    assert_almost_equal(
        multiple_to_multiple_result, compute_pairwise_distance(x, y, metric=name)
    )
    assert_almost_equal(
        multiple_to_multiple_result,
        compute_pairwise_distance(x, y, metric=distance),
    )

    x = _make_3d_series(x)
    y = _make_3d_series(y)

    matrix = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(y)):
            curr_y = y[j]
            matrix[i, j] = distance(curr_x, curr_y)

    assert np.allclose(matrix, multiple_to_multiple_result)

    # If unequal length swap where x and y are to ensure it works both ways around
    if x.shape[-1] != y.shape[-1] and not recursive_call:
        _validate_multiple_to_multiple_result(
            original_y,
            original_x,
            name,
            distance,
            multiple_to_multiple_distance,
            recursive_call=True,
        )


def _validate_single_to_multiple_result(
    x,
    y,
    name,
    distance,
    single_to_multiple_distance,
):
    single_to_multiple_result = single_to_multiple_distance(x, y)

    expected_size = len(y)
    if y.ndim == 1:
        expected_size = 1

    assert isinstance(single_to_multiple_result, np.ndarray)
    assert single_to_multiple_result.shape[-1] == expected_size
    assert_almost_equal(
        single_to_multiple_result, compute_pairwise_distance(x, y, metric=name)
    )
    assert_almost_equal(
        single_to_multiple_result, compute_pairwise_distance(x, y, metric=distance)
    )

    for i in range(single_to_multiple_result.shape[-1]):
        curr_y = y[i]
        curr = single_to_multiple_result[0, i]

        curr_x = x
        if curr_x.ndim > curr_y.ndim:
            curr_y = curr_y.reshape((1, curr_y.shape[0]))
        elif curr_x.ndim < curr_y.ndim:
            curr_x = curr_x.reshape((1, curr_x.shape[0]))

        dist = distance(curr_x, curr_y)
        assert_almost_equal(dist, curr)


@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance(dist):
    """Test pairwise distance function."""

    # ================== Test equal length ==================
    # Test collection of univariate time series in the shape (n_cases, n_timepoints)
    _validate_pairwise_result(
        make_example_2d_numpy(5, 5, random_state=1, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test collection of univariate time series in the shape
    # (n_cases, n_channels, n_timepoints)
    _validate_pairwise_result(
        make_example_3d_numpy(5, 1, 5, random_state=1, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test collection of multivariate time series in the shape
    # (n_cases, n_channels, n_timepoints)
    _validate_pairwise_result(
        make_example_3d_numpy(5, 5, 5, random_state=1, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # TODO: Uncomment in PR #1287
    # # ================== Test unequal length ==================
    # # Test collection of unequal length univariate time series in the shape
    # # (n_cases, n_timepoints)
    # _validate_pairwise_result(
    #     make_example_2d_unequal_length(5, random_state=1, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test collection of unequal length univariate time series in the shape
    # # (n_cases, n_channels, n_timepoints)
    # _validate_pairwise_result(
    #     make_example_unequal_length(5, 1, random_state=1, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test collection of unequal length multivariate time series in the shape
    # # (n_cases, n_channels, n_timepoints)
    # _validate_pairwise_result(
    #     make_example_unequal_length(5, 5, random_state=1, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )

    # ============== Test single point series ==============
    if dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES:
        # Test singe point univariate of shape (1, 1)
        _validate_pairwise_result(
            np.array([[10.0]]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_multiple_to_multiple_distances(dist):
    """Test multiple to multiple distances."""
    # ================== Test equal length ==================
    # Test passing two singular univariate time series of shape (n_timepoints,) and
    # (n_timepoints,)
    _validate_multiple_to_multiple_result(
        make_series(5, return_numpy=True, random_state=1),
        make_series(5, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing two collections of univariate time series of shape
    # (n_cases, n_timepoints) and (n_cases, n_timepoints)
    _validate_multiple_to_multiple_result(
        make_example_2d_numpy(5, 5, random_state=1, return_y=False),
        make_example_2d_numpy(10, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing two collections of univariate time series of shape
    # (n_cases, 1, n_timepoints) and (n_cases, 1, n_timepoints)
    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 1, 5, random_state=1, return_y=False),
        make_example_3d_numpy(10, 1, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing two collections of multivariate time series of shape
    # (n_cases, n_channels, n_timepoints) and (n_cases, n_channels, n_timepoints)
    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 5, 5, random_state=1, return_y=False),
        make_example_3d_numpy(10, 5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # TODO: Uncomment in PR #1287
    # # ==================== Unequal length tests ====================
    # # Test passing two singular unequal length univariate time series of shape
    # # (n_timepoints,) and (m_timepoints,)
    # _validate_multiple_to_multiple_result(
    #     make_series(5, return_numpy=True, random_state=1),
    #     make_series(3, return_numpy=True, random_state=2),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test passing two collections of unequal length univariate time series of shape
    # # (n_cases, n_timepoints) and (n_cases, m_timepoints)
    # _validate_multiple_to_multiple_result(
    #     make_example_2d_unequal_length(5, random_state=1, return_y=False),
    #     make_example_2d_unequal_length(10, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test passing two collections of unequal length univariate time series of shape
    # # (n_cases, 1, n_timepoints) and (n_cases, 1, m_timepoints)
    # _validate_multiple_to_multiple_result(
    #     make_example_unequal_length(5, 1, random_state=1, return_y=False),
    #     make_example_unequal_length(10, 1, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test passing two collections of unequal length  multivariate time series of
    # shape (n_cases, n_channels, m_timepoints) and (n_cases, n_channels, n_timepoints)
    # _validate_multiple_to_multiple_result(
    #     make_example_unequal_length(5, 5, random_state=1, return_y=False),
    #     make_example_unequal_length(10, 5, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )

    # ============== Test single point series ==============
    if dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES:
        # Test singe point univariate of shape (1,)
        _validate_multiple_to_multiple_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test singe point univariate of shape (1, 1)
        _validate_multiple_to_multiple_result(
            np.array([[10.0]]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_single_to_multiple_distances(dist):
    # ================== Test equal length ==================
    # Test passing a singular univariate time series of shape (n_timepoints,) compared
    # to a collection of univariate time series of shape (n_cases, n_timepoints)
    _validate_single_to_multiple_result(
        make_series(5, return_numpy=True, random_state=1),
        make_example_2d_numpy(5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular univariate time series of shape (1, n_timepoints) compared
    # to a collection of univariate time series of shape (n_cases, n_timepoints)
    _validate_single_to_multiple_result(
        make_series(5, 1, return_numpy=True, random_state=1),
        make_example_2d_numpy(5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular univariate time series of shape (n_timepoints,) compared
    # to a collection of univariate time series of shape (n_cases, 1, n_timepoints)
    _validate_single_to_multiple_result(
        make_series(5, return_numpy=True, random_state=1),
        make_example_3d_numpy(5, 1, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular univariate time series of shape (1, n_timepoints) compared
    # to a collection of univariate time series of shape (n_cases, 1, n_timepoints)
    _validate_single_to_multiple_result(
        make_series(5, 1, return_numpy=True, random_state=1),
        make_example_3d_numpy(5, 1, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular multivariate time series of shape
    # (n_channels, n_timepoints) compared to a collection of multivariate time series of
    # shape (n_cases, n_channels, n_timepoints)
    _validate_single_to_multiple_result(
        make_series(5, 5, return_numpy=True, random_state=1),
        make_example_3d_numpy(5, 5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # TODO: Uncomment in PR #1287
    # # ==================== Unequal length tests ====================
    # # Test passing a singular univariate time series of shape (n_timepoints,) compared
    # # to a collection of unequal length univariate time series of shape
    # # (n_cases, m_timepoints)
    # _validate_single_to_multiple_result(
    #     make_series(5, return_numpy=True, random_state=1),
    #     make_example_2d_unequal_length(5, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test passing a singular univariate time series of shape (1, n_timepoints)
    # compare to a collection of unequal length univariate time series of shape
    # # (n_cases, m_timepoints)
    # _validate_single_to_multiple_result(
    #     make_series(5, 1, return_numpy=True, random_state=1),
    #     make_example_2d_unequal_length(5, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"]
    # )
    #
    # # Test passing a singular univariate time series of shape (n_timepoints,) compared
    # # to a collection of unequal length univariate time series of shape
    # # (n_cases, 1, m_timepoints)
    # _validate_single_to_multiple_result(
    #     make_series(5, return_numpy=True, random_state=1),
    #     make_example_unequal_length(5, 1, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test passing a singular univariate time series of shape (1, n_timepoints)
    # compared to a collection of unequal length univariate time series of shape
    # # (n_cases, 1, m_timepoints)
    # _validate_single_to_multiple_result(
    #     make_series(5, 1, return_numpy=True, random_state=1),
    #     make_example_unequal_length(5, 1, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # # Test passing a singular multivariate time series of shape
    # # (n_channels, n_timepoints) compared to a collection of unequal length
    # # multivariate time series of shape (n_cases, n_channels, m_timepoints)
    # _validate_single_to_multiple_result(
    #     make_series(5, 5, return_numpy=True, random_state=1),
    #     make_example_unequal_length(5, 5, random_state=2, return_y=False),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )

    # ============== Test single point series ==============
    if dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES:
        # Test singe point univariate of shape (1,) compared to a collection of a
        # single univariate time series in the shape (n_cases, 1)
        _validate_single_to_multiple_result(
            np.array([10.0]),
            make_example_2d_numpy(5, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test singe point univariate of shape (1, 1) compared to a collection of a
        # single univariate time series in the shape (n_cases, 1, 1)
        _validate_single_to_multiple_result(
            np.array([[10.0]]),
            make_example_3d_numpy(5, 1, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test a multivariate single point series in the shape (n_channels, 1)
        # compared to a collection of a single multivariate time series in the shape
        # (n_cases, n_channels, 1)
        _validate_single_to_multiple_result(
            make_series(1, 5, return_numpy=True, random_state=1),
            make_example_3d_numpy(5, 5, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )
