import numpy as np
from numba import njit
import math


@njit(cache=True, fastmath=True)
def create_bounding_matrix(x_size: int, y_size: int, window: float = None):
    if window is None or window >= 1:
        return np.zeros((x_size, y_size))
    return _sakoe_chiba_bounding(x_size, y_size, window)


@njit(cache=True, fastmath=True)
def _sakoe_chiba_bounding(x_size: int, y_size: int,
                          radius_percent: float) -> np.ndarray:
    one_percent = min(x_size, y_size) / 100
    radius = math.ceil(((radius_percent * one_percent) * 100))
    bounding_matrix = np.full((x_size, y_size), np.inf)

    smallest_size = min(x_size, y_size)
    largest_size = max(x_size, y_size)

    width = largest_size - smallest_size + radius
    for i in range(smallest_size):
        lower = max(0, i - radius)
        upper = min(largest_size, i + width)
        for j in range(lower, upper):
            bounding_matrix[j, i] = 1.

    return bounding_matrix
