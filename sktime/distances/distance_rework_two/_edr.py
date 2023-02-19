# -*- coding: utf-8 -*-
from typing import Union, List, Tuple

import numpy as np

from sktime.distances.distance_rework_two._base import (
    ElasticDistance,
    ElasticDistanceReturn,
    get_bounding_matrix,
    _convert_2d
)
from sktime.distances.distance_rework_two._squared import _SquaredDistance

squared_distance = _SquaredDistance().distance_factory()


class _EdrDistance(ElasticDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _distance(
        x: np.ndarray,
        y: np.ndarray,
        window: Union[float, None] = None,
        itakura_max_slope: Union[float, None] = None,
        bounding_matrix: Union[np.ndarray, None] = None,
        epsilon: Union[float, None] = None,
        *args
    ) -> ElasticDistanceReturn:
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.zeros((x_size + 1, y_size + 1))

        if np.array_equal(x, y):
            return 0.0, cost_matrix[1:, 1:]

        _epsilon = epsilon
        if epsilon is None:
            _epsilon = max(np.std(x), np.std(y)) / 4

        for i in range(1, x_size + 1):
            for j in range(1, y_size + 1):
                if np.isfinite(bounding_matrix[i - 1, j - 1]):
                    squared_dist = squared_distance(x[:, i - 1], y[:, j - 1])

                    if squared_dist < _epsilon:
                        cost = 0
                    else:
                        cost = 1
                    cost_matrix[i, j] = min(
                        cost_matrix[i - 1, j - 1] + cost,
                        cost_matrix[i - 1, j] + 1,
                        cost_matrix[i, j - 1] + 1,
                    )

        return cost_matrix[-1, -1], cost_matrix[1:, 1:]

    @staticmethod
    def _result_process(result: float, *args):
        return float(result / max(args[0].shape[-1], args[1].shape[-1]))

    @staticmethod
    def _alignment_path(
        x: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray, bounding_matrix, *args
    ) -> List[Tuple]:
        raise NotImplementedError("Alignment path is not supported for EDR distance")
