# -*- coding: utf-8 -*-
"""This file is only called test_new_distances while the module transitions.
Once the transition is complete, this file will be renamed to test_distances.py."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import distance as compute_distance
from aeon.distances._distance import DISTANCES
from aeon.distances.tests._expected_results import _expected_distance_results
from aeon.distances.tests._utils import create_test_distance_numpy


def _validate_distance_result(
    x, y, name, distance, expected_result=10  # This will be used in a later pr
):
    if expected_result is None:
        return

    dist_result = distance(x, y)

    assert isinstance(dist_result, float)
    assert_almost_equal(dist_result, expected_result)
    assert dist_result == compute_distance(x, y, metric=name)
    if name != "lcss" and name != "edr":
        assert dist_result == compute_distance(x, y, metric=distance)

    dist_result_to_self = distance(x, x)
    assert isinstance(dist_result_to_self, float)


@pytest.mark.parametrize("dist", DISTANCES)
def test_distances(dist):
    # Test univariate
    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
        _validate_distance_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            _expected_distance_results[dist["name"]][0],
        )

    _validate_distance_result(
        create_test_distance_numpy(10),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][1],
    )

    # Test multivariate
    _validate_distance_result(
        create_test_distance_numpy(10, 10),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][2],
    )

    # Test unequal length
    _validate_distance_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][3],
    )

    _validate_distance_result(
        create_test_distance_numpy(10, 5),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][4],
    )
