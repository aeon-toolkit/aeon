import numpy as np
from aeon.distance_rework import (
    create_bounding_matrix,
    euclidean_distance,
    squared_distance,
    dtw_distance,
    dtw_cost_matrix,
    ddtw_distance,
    ddtw_cost_matrix,
    wdtw_distance,
    wdtw_cost_matrix,
    wddtw_distance,
    wddtw_cost_matrix,
    edr_distance,
    edr_cost_matrix,
    erp_distance,
    erp_cost_matrix,
    lcss_distance,
    lcss_cost_matrix,
    msm_distance,
    msm_cost_matrix,
    twe_distance,
    twe_cost_matrix,
)

x = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)


def cost_matrix_matches_bounding_matrix(
        cost_matrix, expected_shape, bounding_matrix, out_of_bounds_value=np.inf
):
    """Check that the cost matrix matches the bounding matrix."""
    assert cost_matrix.shape == expected_shape
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            if not bounding_matrix[i, j]:
                assert cost_matrix[i, j] == out_of_bounds_value


def almost_equal(first: float, second: float, less_than_val: float = 0.0001) -> bool:
    """On some hardware decimal place values are not the same (e.g. Apple Silicon).
    This function is used to check if two floats are almost equal.
    Parameters
    ----------
    first : float
        First float to compare.
    second : float
        Second float to compare.
    less_than_val : float, default = 0.0001
        The value that the difference between the two floats must be less than.
    Returns
    -------
    bool
        True if floats are almost equal, False otherwise.
    """
    return abs(first - second) < less_than_val


def test_euclidean_distance():
    assert euclidean_distance(x, y) == np.linalg.norm(x - y)


def test_squared_distance():
    assert squared_distance(x, y) == np.linalg.norm(x - y) ** 2


def test_dtw_distance():
    assert almost_equal(dtw_distance(x, y), 4408.25)
    assert almost_equal(dtw_distance(x, y, window=0.1), 4408.25)
    cost_matrix = dtw_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(cost_matrix, (10, 10), bounding_matrix)


def test_ddtw_distance():
    assert almost_equal(ddtw_distance(x, y), 3833.84375)
    assert almost_equal(ddtw_distance(x, y, window=0.1), 3833.84375)
    cost_matrix = ddtw_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[1] - 2, window=0.2)
    cost_matrix_matches_bounding_matrix(cost_matrix, (8, 8), bounding_matrix)


def test_wdtw_distance():
    assert almost_equal(wdtw_distance(x, y), 1930.035439970180)
    assert almost_equal(wdtw_distance(x, y, window=0.1), 1930.035439970180)
    cost_matrix = wdtw_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(cost_matrix, (10, 10), bounding_matrix)


def test_wddtw_distance():
    assert almost_equal(wddtw_distance(x, y), 1725.86611586604)
    assert almost_equal(wddtw_distance(x, y, window=0.1), 1725.86611586604)
    cost_matrix = wddtw_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[1] - 2, window=0.2)
    cost_matrix_matches_bounding_matrix(cost_matrix, (8, 8), bounding_matrix)


def test_edr_distance():
    assert almost_equal(edr_distance(x, y), 0.8)
    assert almost_equal(edr_distance(x, y, window=0.1), 0.1)
    cost_matrix = edr_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )


def test_erp_distance():
    assert almost_equal(erp_distance(x, y), 4408.25)
    assert almost_equal(erp_distance(x, y, window=0.1), 245.0)
    cost_matrix = erp_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )


def test_lcss_distance():
    assert almost_equal(lcss_distance(x, y), 0.8)
    assert almost_equal(lcss_distance(x, y, window=0.1), 0.8)
    cost_matrix = lcss_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )


def test_msm_distance():
    assert almost_equal(msm_distance(x, y), 164.5)
    assert almost_equal(msm_distance(x, y, window=0.1), 3.0)
    assert almost_equal(msm_distance(x, y, independent=False), 164.0)
    assert almost_equal(msm_distance(x, y, independent=False, window=0.1), 1.0)
    cost_matrix = msm_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    cost_matrix = msm_cost_matrix(x, y, window=0.2, independent=False)
    bounding_matrix = create_bounding_matrix(
        x.shape[1], y.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
def test_lcss_distance():
    assert almost_equal(twe_distance(x, y), 8815.5)
    assert almost_equal(twe_distance(x, y, window=0.1), 1501.001)
    cost_matrix = twe_cost_matrix(x, y, window=0.2)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window=0.2)
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )

