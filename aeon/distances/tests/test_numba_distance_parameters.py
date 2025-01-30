"""Test suite for numba distances with parameters."""

from typing import Callable

import numpy as np
import pytest

from aeon.distances import distance
from aeon.distances._distance import DISTANCES, MIN_DISTANCES, MP_DISTANCES
from aeon.distances.elastic._shape_dtw import _pad_ts_edges, _transform_subsequences
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)
from aeon.testing.expected_results.expected_distance_results import (
    _expected_distance_results_params,
)


def _generate_shape_dtw_params(x: np.ndarray, y: np.ndarray):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    padded_x = _pad_ts_edges(x=x, reach=4)
    padded_y = _pad_ts_edges(x=y, reach=4)

    transformed_x = _transform_subsequences(x=padded_x, reach=4)
    transformed_y = _transform_subsequences(x=padded_y, reach=4)
    return {
        "transformation_precomputed": True,
        "transformed_x": transformed_x,
        "transformed_y": transformed_y,
        "reach": 10,
    }


def _test_distance_params(
    param_list: list[dict], distance_func: Callable, distance_str: str
):
    """
    Test function to check the parameters of distance functions.

    Parameters
    ----------
    param_list (List[Dict]): List of dictionaries,
    containing parameters for the distance function.
    distance_func (Callable): The distance function to be tested.
    distance_str (str): The name of the distance function.
    """
    x_univ = make_example_1d_numpy(10, random_state=1)
    y_univ = make_example_1d_numpy(10, random_state=2)

    x_multi = make_example_2d_numpy_series(10, 10, random_state=1)
    y_multi = make_example_2d_numpy_series(10, 10, random_state=2)

    if distance_str == "shift_scale":
        # Shift it to test the max_shift parameter works
        y_univ = np.roll(x_univ, 4)

    # Shape dtw needs parameters to be generated with the x and y so function used
    if distance_str == "shape_dtw":
        param_list.append(_generate_shape_dtw_params)

    test_ts = [[x_univ, y_univ], [x_multi, y_multi]]
    results_to_fill = []

    i = 0
    for param_dict in param_list:
        callable_param_dict = None
        if isinstance(param_dict, Callable):
            callable_param_dict = param_dict

        g_none = False
        if distance_str == "erp" and "g" in param_dict and param_dict["g"] is None:
            g_none = True

        j = 0
        curr_results = []
        for x, y in test_ts:
            if callable_param_dict is not None:
                param_dict = callable_param_dict(x, y)
            if g_none:
                _x = x
                if x.ndim == 1:
                    _x = x.reshape(1, -1)
                _y = y
                if y.ndim == 1:
                    _y = y.reshape(1, -1)

                param_dict["g_arr"] = np.std([_x, _y], axis=0).sum(axis=1)
                if "g" in param_dict:
                    del param_dict["g"]
            results = [
                distance_func(x, y, **param_dict.copy()),
                distance(x, y, method=distance_str, **param_dict.copy()),
            ]

            if distance_str in _expected_distance_results_params:
                res = []
                if _expected_distance_results_params[distance_str][i][j] is not None:
                    for result in results:
                        res.append(result)
                        assert result == pytest.approx(
                            _expected_distance_results_params[distance_str][i][j]
                        )

            curr_results.append(results[0])
            j += 1
        i += 1
        results_to_fill.append(curr_results)


BASIC_BOUNDING_PARAMS = [
    {"window": 0.2},
    {"itakura_max_slope": 0.2},
]

DIST_PARAMS = {
    "dtw": BASIC_BOUNDING_PARAMS,
    "erp": BASIC_BOUNDING_PARAMS + [{"g": 0.5}, {"g": None}],
    "edr": BASIC_BOUNDING_PARAMS + [{"epsilon": 0.5}],
    "lcss": BASIC_BOUNDING_PARAMS + [{"epsilon": 0.5}],
    "ddtw": BASIC_BOUNDING_PARAMS,
    "wdtw": BASIC_BOUNDING_PARAMS + [{"g": 1.0}],
    "wddtw": BASIC_BOUNDING_PARAMS + [{"g": 1.0}],
    "twe": BASIC_BOUNDING_PARAMS + [{"lmbda": 0.5}, {"nu": 0.9}],
    "msm": BASIC_BOUNDING_PARAMS + [{"independent": False}, {"c": 0.1}],
    "adtw": BASIC_BOUNDING_PARAMS + [{"warp_penalty": 5.0}],
    "minkowski": [{"p": 1.0}, {"p": 2.0}],
    "sbd": [{"standardize": False}],
    "shape_dtw": BASIC_BOUNDING_PARAMS + [{"reach": 4}],
    "shift_scale": [{"max_shift": 1}, {"max_shift": None}],
    "soft_dtw": BASIC_BOUNDING_PARAMS + [{"gamma": 0.2}],
}


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_distance_params(dist):
    """Test function to check the parameters of distance functions."""
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return

    if dist["name"] in DIST_PARAMS:
        _test_distance_params(
            DIST_PARAMS[dist["name"]],
            dist["distance"],
            dist["name"],
        )
