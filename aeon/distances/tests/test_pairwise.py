from typing import List, Union

import numpy as np
import pytest
from numba import TypingError
from numpy.testing import assert_almost_equal

from aeon.distances import pairwise_distance as compute_pairwise_distance
from aeon.distances._distance import DISTANCES
from aeon.distances.tests.test_utils import _make_3d_series
from aeon.testing.utils.data_gen import make_example_3d_numpy, make_series


def _validate_pairwise_result(
    x: Union[np.ndarray, List[np.ndarray]],
    name,
    distance,
    pairwise_distance,
):
    pairwise_result = pairwise_distance(x)

    expected_size = (len(x), len(x))

    assert isinstance(pairwise_result, np.ndarray)
    assert pairwise_result.shape == expected_size
    if isinstance(x, np.ndarray):
        assert_almost_equal(pairwise_result, compute_pairwise_distance(x, metric=name))
        assert_almost_equal(
            pairwise_result, compute_pairwise_distance(x, metric=distance)
        )

        x = _make_3d_series(x)

    matrix = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(x)):
            curr_y = x[j]
            matrix[i, j] = distance(curr_x, curr_y)

    assert np.allclose(matrix, pairwise_result)


def _validate_multiple_to_multiple_result(
    x: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[np.ndarray]],
    name,
    distance,
    multiple_to_multiple_distance,
):
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

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
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


def _supports_nonequal_length(dist) -> bool:
    anns = dist["pairwise_distance"].__annotations__
    return any(param in anns and str(List) in str(anns[param]) for param in ["x", "X"])


@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance(dist):
    """Test pairwise distance function."""
    _validate_pairwise_result(
        make_series(5, 5, return_numpy=True, random_state=1).T,
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_pairwise_result(
        make_example_3d_numpy(5, 1, 5, random_state=1)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_pairwise_result(
        make_example_3d_numpy(5, 5, 5, random_state=1)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # unequal-length
    if _supports_nonequal_length(dist):
        _validate_pairwise_result(
            [make_series(5 + i, random_state=1, return_numpy=True) for i in range(5)],
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        _validate_pairwise_result(
            [
                make_series(5 + i, 1, random_state=1, return_numpy=True).reshape(1, -1)
                for i in range(5)
            ],
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        _validate_pairwise_result(
            [
                make_series(5 + i, 5, random_state=1, return_numpy=True).T
                for i in range(5)
            ],
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_multiple_to_multiple_distances(dist):
    """Test multiple to multiple distances."""
    # Univariate tests
    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
        _validate_multiple_to_multiple_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

    _validate_multiple_to_multiple_result(
        make_series(5, return_numpy=True, random_state=1),
        make_series(5, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Multivariate tests
    _validate_multiple_to_multiple_result(
        make_series(5, 5, return_numpy=True, random_state=1).T,
        make_series(5, 5, return_numpy=True, random_state=2).T,
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Dataset tests
    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 1, 5, random_state=1)[0],
        make_example_3d_numpy(5, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 5, 5, random_state=1)[0],
        make_example_3d_numpy(5, 5, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Unequal length tests
    _validate_multiple_to_multiple_result(
        make_series(5, return_numpy=True, random_state=1),
        make_series(3, return_numpy=True, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        make_series(5, 5, return_numpy=True, random_state=1).T,
        make_series(5, 3, return_numpy=True, random_state=2).T,
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        make_example_3d_numpy(5, 5, 5, random_state=1)[0],
        make_example_3d_numpy(5, 5, 3, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # unequal-length (within dataset)
    if _supports_nonequal_length(dist):
        # univariate
        _validate_multiple_to_multiple_result(
            [make_series(5, random_state=1, return_numpy=True)],
            [make_series(3, random_state=2, return_numpy=True)],
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # multivariate dataset
        _validate_multiple_to_multiple_result(
            [
                make_series(5 + i, 1, random_state=1, return_numpy=True).reshape(1, -1)
                for i in range(5)
            ],
            [
                make_series(3 + i, 1, random_state=2, return_numpy=True).reshape(1, -1)
                for i in range(5)
            ],
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        _validate_multiple_to_multiple_result(
            [
                make_series(5 + i, 5, random_state=1, return_numpy=True).T
                for i in range(5)
            ],
            [
                make_series(3 + i, 5, random_state=2, return_numpy=True).T
                for i in range(5)
            ],
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

        # multivariate with different number of channels
        if dist["name"] != "sbd":
            # - different between x and y
            _validate_multiple_to_multiple_result(
                [
                    make_series(5 + i, 2, random_state=1, return_numpy=True).T
                    for i in range(5)
                ],
                [
                    make_series(3 + i, 3, random_state=2, return_numpy=True).T
                    for i in range(5)
                ],
                dist["name"],
                dist["distance"],
                dist["pairwise_distance"],
            )
            # TypedList does not support array(float64, 1d, C) mixed with
            # array(float64, 2d, C) or other dimensions
            with pytest.raises(TypingError):
                # - different just in x
                _validate_multiple_to_multiple_result(
                    [
                        make_series(
                            5 + i, 1 + i % 2, random_state=1, return_numpy=True
                        ).T
                        for i in range(5)
                    ],
                    [
                        make_series(3 + i, 1, random_state=2, return_numpy=True).T
                        for i in range(5)
                    ],
                    dist["name"],
                    dist["distance"],
                    dist["pairwise_distance"],
                )
            with pytest.raises(TypingError):
                # - different just in y
                _validate_multiple_to_multiple_result(
                    [
                        make_series(5 + i, 1, random_state=1, return_numpy=True).T
                        for i in range(5)
                    ],
                    [
                        make_series(
                            3 + i, 1 + i % 2, random_state=2, return_numpy=True
                        ).T
                        for i in range(5)
                    ],
                    dist["name"],
                    dist["distance"],
                    dist["pairwise_distance"],
                )
            with pytest.raises(TypingError):
                # - different in both
                _validate_multiple_to_multiple_result(
                    [
                        make_series(
                            5 + i, 1 + i % 2, random_state=1, return_numpy=True
                        ).T
                        for i in range(5)
                    ],
                    [
                        make_series(
                            3 + i, 1 + i % 2, random_state=2, return_numpy=True
                        ).T
                        for i in range(5)
                    ],
                    dist["name"],
                    dist["distance"],
                    dist["pairwise_distance"],
                )


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_single_to_multiple_distances(dist):
    # Univariate tests
    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
        _validate_single_to_multiple_result(
            np.array([10.0]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

    _validate_single_to_multiple_result(
        make_series(5, return_numpy=True, random_state=1),
        make_example_3d_numpy(3, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        make_series(5, 1, return_numpy=True, random_state=1).T,
        make_example_3d_numpy(5, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Multivariate tests
    _validate_single_to_multiple_result(
        make_series(5, 5, return_numpy=True, random_state=1).T,
        make_example_3d_numpy(5, 5, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Unequal length tests
    _validate_single_to_multiple_result(
        make_series(3, return_numpy=True, random_state=1),
        make_example_3d_numpy(3, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        make_series(3, return_numpy=True, random_state=1),
        make_example_3d_numpy(3, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        make_series(3, 5, return_numpy=True, random_state=1).T,
        make_example_3d_numpy(5, 5, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        make_series(5, 5, return_numpy=True, random_state=1).T,
        make_example_3d_numpy(5, 5, 3, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )
