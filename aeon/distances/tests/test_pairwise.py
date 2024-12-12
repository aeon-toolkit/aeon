"""Test for pairwise distances."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import pairwise_distance as compute_pairwise_distance
from aeon.distances._distance import (
    DISTANCES,
    MIN_DISTANCES,
    MP_DISTANCES,
    SINGLE_POINT_NOT_SUPPORTED_DISTANCES,
    SYMMETRIC_DISTANCES,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_list,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)


def _make_3d_series(x: np.ndarray) -> np.ndarray:
    n_channels = x.ndim
    if n_channels == 1:
        shape = x.shape
        _x = np.reshape(x, (1, 1, shape[0]))
    elif n_channels == 2:
        shape = x.shape
        _x = np.reshape(x, (shape[0], 1, shape[1]))
    elif n_channels > 3:
        raise ValueError(
            "The matrix provided has more than 3 dimensions. This is not"
            "supported. Please provide a matrix with less than "
            "3 dimensions"
        )
    else:
        _x = x
    return _x


def _validate_pairwise_result(
    x: np.ndarray,
    name,
    distance,
    pairwise_distance,
):
    """Validate pairwise result.

    Parameters
    ----------
    x: Input np.ndarray.
    name: Name of the distance method.
    distance: Distance function.
    pairwise_distance: Pairwise distance function.
    """
    symmetric = name in SYMMETRIC_DISTANCES
    pairwise_result = pairwise_distance(x)

    expected_size = (len(x), len(x))

    assert isinstance(pairwise_result, np.ndarray)
    assert pairwise_result.shape == expected_size
    assert_almost_equal(
        pairwise_result, compute_pairwise_distance(x, method=name, symmetric=symmetric)
    )
    assert_almost_equal(
        pairwise_result,
        compute_pairwise_distance(x, method=distance, symmetric=symmetric),
    )

    if isinstance(x, np.ndarray):
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
    check_xy_permuted=True,
):
    """
    Validate multiple to multiple result.

    Parameters
    ----------
    x: Input array.
    y: Input array.
    name: Name of the distance method.
    distance: Distance function.
    multiple_to_multiple_distance: Mul-to-Mul distance function.
    check_xy_permuted: recursively call with swapped series
    """
    original_x = x.copy()
    original_y = y.copy()
    multiple_to_multiple_result = multiple_to_multiple_distance(x, y)

    if (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and isinstance(y, np.ndarray)
        and y.ndim == 1
    ):
        expected_size = (1, 1)
    else:
        expected_size = (len(x), len(y))

    assert isinstance(multiple_to_multiple_result, np.ndarray)
    assert multiple_to_multiple_result.shape == expected_size

    assert_almost_equal(
        multiple_to_multiple_result, compute_pairwise_distance(x, y, method=name)
    )
    assert_almost_equal(
        multiple_to_multiple_result,
        compute_pairwise_distance(x, y, method=distance),
    )

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
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
    if check_xy_permuted and (
        isinstance(x, list) or isinstance(y, list) or x.shape[-1] != y.shape[-1]
    ):
        _validate_multiple_to_multiple_result(
            original_y,
            original_x,
            name,
            distance,
            multiple_to_multiple_distance,
            check_xy_permuted=False,
        )


def _validate_single_to_multiple_result(
    x, y, name, distance, single_to_multiple_distance
):
    """
    Validate single to multiple result.

    Parameters
    ----------
    x: Input array.
    y: Input array.
    name: Name of the distance method.
    distance: Distance function.
    single_to_multiple_distance: Single to multiple distance function.
    run_inverse: Boolean that reruns the test with x and y swapped in position
    """
    symmetric = name in SYMMETRIC_DISTANCES
    single_to_multiple_result = single_to_multiple_distance(x, y)

    expected_size = len(y)
    if isinstance(y, np.ndarray) and y.ndim == 1:
        expected_size = 1

    assert isinstance(single_to_multiple_result, np.ndarray)

    if isinstance(x, np.ndarray):
        x_shape = x.shape
    else:
        x_shape = (len(x), *x[0].shape)

    if isinstance(y, np.ndarray):
        y_shape = y.shape
    else:
        y_shape = (len(y), *y[0].shape)

    if len(x_shape) > len(y_shape):
        assert single_to_multiple_result.shape[0] == expected_size
    else:
        assert single_to_multiple_result.shape[1] == expected_size
    assert_almost_equal(
        single_to_multiple_result,
        compute_pairwise_distance(x, y, method=name, symmetric=symmetric),
    )
    assert_almost_equal(
        single_to_multiple_result,
        compute_pairwise_distance(x, y, method=distance, symmetric=symmetric),
    )

    if len(x_shape) < len(y_shape):
        x, y = y, x

    if not isinstance(x, np.ndarray):
        y = np.array(y)

    for i in range(single_to_multiple_result.shape[-1]):
        curr = single_to_multiple_result[0, i]
        curr_x = x[i]
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if curr_x.ndim == 1:
            curr_x = curr_x.reshape(1, -1)

        if symmetric:
            dist = distance(curr_x, y)
        else:
            dist = distance(y, curr_x)
        assert_almost_equal(dist, curr)


def _supports_nonequal_length(dist) -> bool:
    anns = dist["pairwise_distance"].__annotations__
    return any(
        param in anns and str(list) in str(anns[param])
        for param in ["x", "X", "y", "Y"]
    )


@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance(dist):
    """Test pairwise distance function."""
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    # ================== Test equal length ==================
    # Test collection of univariate time series in the shape (n_cases, n_timepoints)
    _validate_pairwise_result(
        make_example_2d_numpy_collection(5, 5, random_state=1, return_y=False),
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

    # ================== Test unequal length ==================
    if _supports_nonequal_length(dist):
        # Test collection of unequal length univariate time series in the shape
        # (n_cases, n_timepoints)
        _validate_pairwise_result(
            make_example_2d_numpy_list(5, random_state=1, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test collection of unequal length univariate time series in the shape
        # (n_cases, n_channels, n_timepoints)
        _validate_pairwise_result(
            make_example_3d_numpy_list(5, 1, random_state=1, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test collection of unequal length multivariate time series in the shape
        # (n_cases, n_channels, n_timepoints)
        _validate_pairwise_result(
            make_example_3d_numpy_list(5, 5, random_state=1, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

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
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    # ================== Test equal length ==================
    # Test passing two singular univariate time series of shape (n_timepoints,)
    if dist["name"] != "scale_shift":
        _validate_multiple_to_multiple_result(
            make_example_1d_numpy(5, random_state=1),
            make_example_1d_numpy(5, random_state=2),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

    # Test passing two collections of univariate time series of shape
    # (n_cases, n_timepoints)
    _validate_multiple_to_multiple_result(
        make_example_2d_numpy_collection(5, 5, random_state=1, return_y=False),
        make_example_2d_numpy_collection(10, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing two collections of univariate time series of shape
    # (n_cases, 1, n_timepoints)
    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 1, 5, random_state=1, return_y=False),
        make_example_3d_numpy(10, 1, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing two collections of multivariate time series of shape
    # (n_cases, n_channels, n_timepoints)
    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 5, 5, random_state=1, return_y=False),
        make_example_3d_numpy(10, 5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # ==================== Unequal length tests ====================
    if _supports_nonequal_length(dist):
        # Test passing two singular unequal length univariate time series of shape
        # (n_timepoints,) and (m_timepoints,)
        _validate_multiple_to_multiple_result(
            make_example_1d_numpy(5, random_state=1),
            make_example_1d_numpy(3, random_state=2),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test passing two collections of unequal length univariate time series of shape
        # (n_cases, n_timepoints) and (n_cases, m_timepoints)
        _validate_multiple_to_multiple_result(
            make_example_2d_numpy_list(5, random_state=1, return_y=False),
            make_example_2d_numpy_list(10, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test passing two collections of unequal length univariate time series of shape
        # (n_cases, 1, n_timepoints) and (n_cases, 1, m_timepoints)
        _validate_multiple_to_multiple_result(
            make_example_3d_numpy_list(5, 1, random_state=1, return_y=False),
            make_example_3d_numpy_list(10, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test passing two collections of unequal length  multivariate time series of
        # shape (n_cases, n_channels, m_timepoints) and (n_cases, n_channels,
        # n_timepoints)
        _validate_multiple_to_multiple_result(
            make_example_3d_numpy_list(5, 5, random_state=1, return_y=False),
            make_example_3d_numpy_list(10, 5, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

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
def test_single_to_multiple_distances(dist):
    """Test single to multiple distances."""
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    # ================== Test equal length ==================
    # Test passing a singular univariate time series of shape (n_timepoints,) compared
    # to a collection of univariate time series of shape (n_cases, n_timepoints)
    _validate_single_to_multiple_result(
        make_example_1d_numpy(5, random_state=1),
        make_example_2d_numpy_collection(5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular univariate time series of shape (n_timepoints,) compared
    # to a collection of univariate time series of shape (n_cases, 1, n_timepoints)
    _validate_single_to_multiple_result(
        make_example_1d_numpy(5, random_state=1),
        make_example_3d_numpy(5, 1, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular univariate time series of shape (1, n_timepoints) compared
    # to a collection of univariate time series of shape (n_cases, 1, n_timepoints)
    _validate_single_to_multiple_result(
        make_example_2d_numpy_series(5, 1, random_state=1),
        make_example_3d_numpy(5, 1, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Test passing a singular multivariate time series of shape
    # (n_channels, n_timepoints) compared to a collection of multivariate time series
    # of shape (n_cases, n_channels, n_timepoints)
    _validate_single_to_multiple_result(
        make_example_2d_numpy_series(5, 5, random_state=1),
        make_example_3d_numpy(5, 5, 5, random_state=2, return_y=False),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # ==================== Unequal length tests ====================
    if _supports_nonequal_length(dist):
        # Test passing a singular univariate time series of shape (n_timepoints,)
        # compared to a collection of unequal length univariate time series of shape
        # (n_cases, m_timepoints)
        _validate_single_to_multiple_result(
            make_example_1d_numpy(5, random_state=1),
            make_example_2d_numpy_list(5, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test passing a singular univariate time series of shape (1, n_timepoints)
        # compare to a collection of unequal length univariate time series of shape
        # (n_cases, m_timepoints)
        _validate_single_to_multiple_result(
            make_example_2d_numpy_series(5, 1, random_state=1),
            make_example_2d_numpy_list(5, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test passing a singular univariate time series of shape (n_timepoints,)
        # compared to a collection of unequal length univariate time series of shape
        # (n_cases, 1, m_timepoints)
        _validate_single_to_multiple_result(
            make_example_1d_numpy(5, random_state=1),
            make_example_3d_numpy_list(5, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # Test passing a singular univariate time series of shape (1, n_timepoints)
        # compared to a collection of unequal length univariate time series of shape
        # (n_cases, 1, m_timepoints)
        _validate_single_to_multiple_result(
            make_example_2d_numpy_series(5, 1, random_state=1),
            make_example_3d_numpy_list(5, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

    # ============== Test single point series ==============
    if dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES:
        # Test singe point univariate of shape (1,) compared to a collection of a
        # single univariate time series in the shape (n_cases, 1)
        _validate_single_to_multiple_result(
            np.array([10.0]),
            make_example_2d_numpy_collection(5, 1, random_state=2, return_y=False),
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
            make_example_2d_numpy_collection(5, 1, random_state=1, return_y=False),
            make_example_3d_numpy(5, 5, 1, random_state=2, return_y=False),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )


@pytest.mark.parametrize("seed", [1, 10, 42, 52, 100])
@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance_non_negative(dist, seed):
    """Most estimators require distances to be non-negative."""
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    X = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, random_state=seed, return_y=False
    )
    X2 = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=10, random_state=seed + 1, return_y=False
    )
    pairwise = dist["pairwise_distance"]
    Xt2 = pairwise(X2, X)
    assert Xt2.min() >= 0, f"Distance {dist['name']} is negative"
