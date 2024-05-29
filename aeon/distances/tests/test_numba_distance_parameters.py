"""Test suite for numba distances with parameters."""

from typing import Callable, Dict, List

import numpy as np
import pytest

from aeon.distances import distance
from aeon.distances._distance import DISTANCES
from aeon.distances.tests.test_utils import _generate_shape_dtw_params
from aeon.testing.expected_results.expected_distance_results import (
    _expected_distance_results_params,
)
from aeon.testing.utils.data_gen import make_series


def _test_distance_params(
    param_list: List[Dict], distance_func: Callable, distance_str: str
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
    x_univ = make_series(10, return_numpy=True, random_state=1)
    y_univ = make_series(10, return_numpy=True, random_state=2)

    x_multi = make_series(10, 10, return_numpy=True, random_state=1)
    y_multi = make_series(10, 10, return_numpy=True, random_state=2)

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
                distance(x, y, metric=distance_str, **param_dict.copy()),
            ]

            if distance_str in _expected_distance_results_params:
                if _expected_distance_results_params[distance_str][i][j] is not None:
                    for result in results:
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
}


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_distance_params(dist):
    """Test function to check the parameters of distance functions."""
    if dist["name"] in DIST_PARAMS:
        _test_distance_params(
            DIST_PARAMS[dist["name"]],
            dist["distance"],
            dist["name"],
        )
