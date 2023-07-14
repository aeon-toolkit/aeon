# -*- coding: utf-8 -*-
"""Test suite for numba distances with parameters."""
from typing import Callable, Dict, List

import numpy as np
import pytest

from aeon.distances import distance
from aeon.distances._distance import DISTANCES
from aeon.distances.tests._expected_results import _expected_distance_results_params
from aeon.distances.tests._utils import create_test_distance_numpy


def _test_distance_params(
    param_list: List[Dict], distance_func: Callable, distance_str: str
):
    x_univ = create_test_distance_numpy(10, 1).reshape((1, 10))
    y_univ = create_test_distance_numpy(10, 1, random_state=2).reshape((1, 10))

    x_multi = create_test_distance_numpy(10, 10)
    y_multi = create_test_distance_numpy(10, 10, random_state=2)

    test_ts = [[x_univ, y_univ], [x_multi, y_multi]]
    results_to_fill = []

    i = 0
    for param_dict in param_list:
        g_none = False
        if distance_str == "erp" and "g" in param_dict and param_dict["g"] is None:
            g_none = True

        j = 0
        curr_results = []
        for x, y in test_ts:
            if g_none:
                param_dict["g_arr"] = np.std([x, y], axis=0).sum(axis=1)
                if "g" in param_dict:
                    del param_dict["g"]
            results = []
            results.append(distance_func(x, y, **param_dict))
            results.append(distance(x, y, metric=distance_str, **param_dict))

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
    "msm": BASIC_BOUNDING_PARAMS + [{"independent": False}, {"c": 0.2}],
}


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_distance_params(dist):
    if dist["name"] in DIST_PARAMS:
        _test_distance_params(
            DIST_PARAMS[dist["name"]],
            dist["distance"],
            dist["name"],
        )
