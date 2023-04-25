# -*- coding: utf-8 -*-
"""Test suite for numba distances with parameters."""
from typing import Callable, Dict, List

import numpy as np
import pytest
from numba import njit

from aeon.distances import distance, distance_factory
from aeon.distances._distance import _METRIC_INFOS, NEW_DISTANCES
from aeon.distances._numba_utils import to_numba_timeseries
from aeon.distances.base import MetricInfo
from aeon.distances.tests._expected_results import _expected_distance_results_params
from aeon.distances.tests._utils import create_test_distance_numpy
from aeon.distances.tests.test_new_distances import DISTANCES


def _test_distance_params(
    param_list: List[Dict], distance_func: Callable, distance_str: str
):
    x_univ = to_numba_timeseries(create_test_distance_numpy(10, 1))
    y_univ = to_numba_timeseries(create_test_distance_numpy(10, 1, random_state=2))

    x_multi = create_test_distance_numpy(10, 10)
    y_multi = create_test_distance_numpy(10, 10, random_state=2)

    test_ts = [[x_univ, y_univ], [x_multi, y_multi]]
    results_to_fill = []

    i = 0
    for param_dict in param_list:
        j = 0
        curr_results = []
        for x, y in test_ts:
            results = []
            curr_dist_fact = distance_factory(x, y, metric=distance_str, **param_dict)
            results.append(distance_func(x, y, **param_dict))
            results.append(distance(x, y, metric=distance_str, **param_dict))
            if distance_str not in NEW_DISTANCES:
                results.append(curr_dist_fact(x, y))
            else:
                results.append(curr_dist_fact(x, y, **param_dict))
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

    if distance_str == "erp":
        param_dict = {"g": np.array([0.5])}
        first_uni = distance_func(x_univ, y_univ, **param_dict)
        second_uni = distance(x_univ, y_univ, metric=distance_str, **param_dict)
        param_dict = {"g": np.array(list(range(10)))}
        first_multi = distance_func(x_multi, y_multi, **param_dict)
        second_multi = distance(x_multi, y_multi, metric=distance_str, **param_dict)
        assert first_uni == pytest.approx(second_uni)
        assert first_multi == pytest.approx(second_multi)




BASIC_BOUNDING_PARAMS = [
    {"window": 0.2},
]


@njit(cache=True)
def _test_derivative(q: np.ndarray):
    return q


DIST_PARAMS = {
    "dtw": BASIC_BOUNDING_PARAMS,
    "erp": BASIC_BOUNDING_PARAMS + [{"g": 0.5}],
    "edr": BASIC_BOUNDING_PARAMS + [{"epsilon": 0.5}],
    "lcss": BASIC_BOUNDING_PARAMS + [{"epsilon": 0.5}],
    "ddtw": BASIC_BOUNDING_PARAMS,
    "wdtw": BASIC_BOUNDING_PARAMS + [{"g": 1.0}],
    "wddtw": BASIC_BOUNDING_PARAMS + [{"g": 1.0}],
    "twe": BASIC_BOUNDING_PARAMS + [{"lmbda": 0.5}, {"nu": 0.9}, {"p": 4}],
}


@pytest.mark.parametrize("dist", _METRIC_INFOS)
def test_distance_params(dist: MetricInfo):
    """Test parametisation of distance callables."""
    if dist.canonical_name in DIST_PARAMS:
        _test_distance_params(
            DIST_PARAMS[dist.canonical_name],
            dist.dist_func,
            dist.canonical_name,
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_distance_params(dist):
    if dist["name"] in DIST_PARAMS:
        _test_distance_params(
            DIST_PARAMS[dist["name"]],
            dist["distance"],
            dist["name"],
        )
