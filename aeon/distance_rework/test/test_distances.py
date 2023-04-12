import numpy as np
from aeon.distance_rework import (
    euclidean_distance,
    squared_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
)

x = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)


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


def test_ddtw_distance():
    assert almost_equal(ddtw_distance(x, y), 3833.84375)
    assert almost_equal(ddtw_distance(x, y, window=0.1), 3833.84375)


def test_wdtw_distance():
    assert almost_equal(wdtw_distance(x, y), 1930.035439970180)
    assert almost_equal(wdtw_distance(x, y, window=0.1), 1930.035439970180)
