# -*- coding: utf-8 -*-
"""This file is only called test_new_distances while the module transitions.
Once the transition is complete, this file will be renamed to test_distances.py."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_from_multiple_to_multiple_distance,
    ddtw_from_single_to_multiple_distance,
    ddtw_pairwise_distance,
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_from_multiple_to_multiple_distance,
    dtw_from_single_to_multiple_distance,
    dtw_pairwise_distance,
    euclidean_distance,
    euclidean_from_multiple_to_multiple_distance,
    euclidean_from_single_to_multiple_distance,
    euclidean_pairwise_distance,
    squared_distance,
    squared_from_multiple_to_multiple_distance,
    squared_from_single_to_multiple_distance,
    squared_pairwise_distance,
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_from_multiple_to_multiple_distance,
    wddtw_from_single_to_multiple_distance,
    wddtw_pairwise_distance,
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_from_multiple_to_multiple_distance,
    wdtw_from_single_to_multiple_distance,
    wdtw_pairwise_distance,
    lcss_distance,
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_pairwise_distance,
    lcss_from_multiple_to_multiple_distance,
    lcss_from_single_to_multiple_distance,
)
from aeon.distances.tests._expected_results import _expected_distance_results
from aeon.distances.tests._utils import create_test_distance_numpy

DISTANCES = [
    {
        "name": "euclidean",
        "distance": euclidean_distance,
        "pairwise_distance": euclidean_pairwise_distance,
        "single_to_multiple_distance": euclidean_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": euclidean_from_multiple_to_multiple_distance,
    },
    {
        "name": "squared",
        "distance": squared_distance,
        "pairwise_distance": squared_pairwise_distance,
        "single_to_multiple_distance": squared_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": squared_from_multiple_to_multiple_distance,
    },
    {
        "name": "dtw",
        "distance": dtw_distance,
        "pairwise_distance": dtw_pairwise_distance,
        "single_to_multiple_distance": dtw_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": dtw_from_multiple_to_multiple_distance,
        "cost_matrix": dtw_cost_matrix,
        "alignment_path": dtw_alignment_path,
    },
    {
        "name": "ddtw",
        "distance": ddtw_distance,
        "pairwise_distance": ddtw_pairwise_distance,
        "single_to_multiple_distance": ddtw_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": ddtw_from_multiple_to_multiple_distance,
        "cost_matrix": ddtw_cost_matrix,
        "alignment_path": ddtw_alignment_path,
    },
    {
        "name": "wdtw",
        "distance": wdtw_distance,
        "pairwise_distance": wdtw_pairwise_distance,
        "single_to_multiple_distance": wdtw_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": wdtw_from_multiple_to_multiple_distance,
        "cost_matrix": wdtw_cost_matrix,
        "alignment_path": wdtw_alignment_path,
    },
    {
        "name": "wddtw",
        "distance": wddtw_distance,
        "pairwise_distance": wddtw_pairwise_distance,
        "single_to_multiple_distance": wddtw_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": wddtw_from_multiple_to_multiple_distance,
        "cost_matrix": wddtw_cost_matrix,
        "alignment_path": wddtw_alignment_path,
    },
    {
        "name": "lcss",
        "distance": lcss_distance,
        "pairwise_distance": lcss_pairwise_distance,
        "single_to_multiple_distance": lcss_from_single_to_multiple_distance,
        "multiple_to_multiple_distance": lcss_from_multiple_to_multiple_distance,
        "cost_matrix": lcss_cost_matrix,
        "alignment_path": lcss_alignment_path,
    }
]


def _validate_distance_result(
    x, y, name, distance, expected_result=10  # This will be used in a later pr
):
    if expected_result is None:
        return

    dist_result = distance(x, y)

    assert isinstance(dist_result, float)
    assert_almost_equal(dist_result, expected_result)

    dist_result_to_self = distance(x, x)
    assert isinstance(dist_result_to_self, float)


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_distances(dist):
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

    _validate_distance_result(
        create_test_distance_numpy(2, 1, 10)[0],
        create_test_distance_numpy(2, 1, 10, random_state=2)[0],
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
    if len(_expected_distance_results[dist["name"]]) < 3:
        _expected_distance_results[dist["name"]] = list(range(0, 10))

    _validate_distance_result(
        create_test_distance_numpy(10, 10, 10),
        create_test_distance_numpy(10, 10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][3],
    )

    # Test unequal length
    _validate_distance_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][4],
    )

    _validate_distance_result(
        create_test_distance_numpy(10, 5),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][5],
    )

    _validate_distance_result(
        create_test_distance_numpy(10, 10, 5),
        create_test_distance_numpy(10, 10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][6],
    )
