import numpy as np
from aeon.distance_rework import (
    euclidean_distance,
    squared_distance
)


def test_euclidean_distance():
    x = np.array([[1, 2, 3]])
    y = np.array([[4, 5, 6]])
    assert euclidean_distance(x, y) == np.linalg.norm(x - y)


def test_squared_distance():
    x = np.array([[1, 2, 3]])
    y = np.array([[4, 5, 6]])
    assert squared_distance(x, y) == np.linalg.norm(x - y) ** 2


