# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import numpy as np

from aeon.distances import (
    create_bounding_matrix,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_from_multiple_to_multiple_distance,
    ddtw_from_single_to_multiple_distance,
    ddtw_pairwise_distance,
    dtw_cost_matrix,
    dtw_distance,
    dtw_from_multiple_to_multiple_distance,
    dtw_from_single_to_multiple_distance,
    dtw_pairwise_distance,
    edr_cost_matrix,
    edr_distance,
    edr_from_multiple_to_multiple_distance,
    edr_from_single_to_multiple_distance,
    edr_pairwise_distance,
    erp_cost_matrix,
    erp_distance,
    erp_from_multiple_to_multiple_distance,
    erp_from_single_to_multiple_distance,
    erp_pairwise_distance,
    euclidean_distance,
    euclidean_from_multiple_to_multiple_distance,
    euclidean_from_single_to_multiple_distance,
    euclidean_pairwise_distance,
    lcss_cost_matrix,
    lcss_distance,
    lcss_from_multiple_to_multiple_distance,
    lcss_from_single_to_multiple_distance,
    lcss_pairwise_distance,
    msm_cost_matrix,
    msm_distance,
    msm_from_multiple_to_multiple_distance,
    msm_from_single_to_multiple_distance,
    msm_pairwise_distance,
    squared_distance,
    squared_from_multiple_to_multiple_distance,
    squared_from_single_to_multiple_distance,
    squared_pairwise_distance,
    twe_cost_matrix,
    twe_distance,
    twe_from_multiple_to_multiple_distance,
    twe_from_single_to_multiple_distance,
    twe_pairwise_distance,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_from_multiple_to_multiple_distance,
    wddtw_from_single_to_multiple_distance,
    wddtw_pairwise_distance,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_from_multiple_to_multiple_distance,
    wdtw_from_single_to_multiple_distance,
    wdtw_pairwise_distance,
)
from aeon.distances.tests._utils import create_test_distance_numpy

x_univariate = np.array([[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13]])
y_univariate = np.array([[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14]])

x_multivariate = np.array(
    [[2, 35, 14, 5, 68, 7.5, 68, 7, 11, 13], [5, 68, 7.5, 68, 7, 11, 13, 5, 68, 7]]
)
y_multivariate = np.array(
    [[8, 19, 10, 12, 68, 7.5, 60, 7, 10, 14], [15, 12, 4, 62, 17, 10, 3, 15, 48, 7]]
)


def _additional_distance_tests(
    x,
    y,
    pairwise_distance,
    single_to_multiple_distance,
    multiple_to_multiple_distance,
    distance_function,
):
    assert pairwise_distance.shape == (5, 5)
    assert single_to_multiple_distance.shape == (5,)
    assert multiple_to_multiple_distance.shape == (5, 5)

    for i in range(pairwise_distance.shape[0]):
        for j in range(pairwise_distance.shape[0]):
            np.testing.assert_almost_equal(
                pairwise_distance[i, j], multiple_to_multiple_distance[i, j]
            )
            np.testing.assert_almost_equal(
                pairwise_distance[i, j], distance_function(x[i], x[j])
            )

    for i in range(single_to_multiple_distance.shape[0]):
        np.testing.assert_almost_equal(
            single_to_multiple_distance[i], distance_function(x[0], y[i])
        )


def cost_matrix_matches_bounding_matrix(
    cost_matrix, expected_shape, bounding_matrix, out_of_bounds_value=np.inf
):
    """Check that the cost matrix matches the bounding matrix."""
    assert cost_matrix.shape == expected_shape
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            if not bounding_matrix[i, j]:
                assert cost_matrix[i, j] == out_of_bounds_value


def test_euclidean_distance():
    assert euclidean_distance(x_multivariate, y_multivariate) == np.linalg.norm(
        x_multivariate - y_multivariate
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = euclidean_pairwise_distance(x)
    single_to_multiple_distance = euclidean_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = euclidean_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        euclidean_distance,
    )


def test_squared_distance():
    assert (
        squared_distance(x_multivariate, y_multivariate)
        == np.linalg.norm(x_multivariate - y_multivariate) ** 2
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = squared_pairwise_distance(x)
    single_to_multiple_distance = squared_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = squared_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        squared_distance,
    )


def test_dtw_distance():
    np.testing.assert_almost_equal(
        dtw_distance(x_multivariate, y_multivariate), 4408.25
    )
    np.testing.assert_almost_equal(
        dtw_distance(x_multivariate, y_multivariate, window=0.1), 4408.25
    )
    cost_matrix = dtw_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(cost_matrix, (10, 10), bounding_matrix)
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = dtw_pairwise_distance(x)
    single_to_multiple_distance = dtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = dtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        dtw_distance,
    )


def test_ddtw_distance():
    np.testing.assert_almost_equal(ddtw_distance(x_univariate, y_univariate), 443.25)
    np.testing.assert_almost_equal(
        wddtw_distance(x_univariate, y_univariate), 199.53608069124422
    )
    np.testing.assert_almost_equal(
        ddtw_distance(x_multivariate, y_multivariate), 3833.84375
    )
    np.testing.assert_almost_equal(
        ddtw_distance(x_multivariate, y_multivariate, window=0.1), 3833.84375
    )
    cost_matrix = ddtw_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1] - 2, y_multivariate.shape[1] - 2, window=0.2
    )
    cost_matrix_matches_bounding_matrix(cost_matrix, (8, 8), bounding_matrix)
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = ddtw_pairwise_distance(x)
    single_to_multiple_distance = ddtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = ddtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        ddtw_distance,
    )


def test_wdtw_distance():
    np.testing.assert_almost_equal(
        wdtw_distance(x_multivariate, y_multivariate), 1930.035439970180
    )
    np.testing.assert_almost_equal(
        wdtw_distance(x_multivariate, y_multivariate, window=0.1), 1930.035439970180
    )
    cost_matrix = wdtw_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(cost_matrix, (10, 10), bounding_matrix)
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = wdtw_pairwise_distance(x)
    single_to_multiple_distance = wdtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = wdtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        wdtw_distance,
    )


def test_wddtw_distance():
    np.testing.assert_almost_equal(
        wddtw_distance(x_multivariate, y_multivariate), 1725.86611586604
    )
    np.testing.assert_almost_equal(
        wddtw_distance(x_multivariate, y_multivariate, window=0.1), 1725.86611586604
    )
    cost_matrix = wddtw_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1] - 2, y_multivariate.shape[1] - 2, window=0.2
    )
    cost_matrix_matches_bounding_matrix(cost_matrix, (8, 8), bounding_matrix)
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = wddtw_pairwise_distance(x)
    single_to_multiple_distance = wddtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = wddtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        wddtw_distance,
    )


def test_edr_distance():
    np.testing.assert_almost_equal(edr_distance(x_multivariate, y_multivariate), 0.8)
    np.testing.assert_almost_equal(
        edr_distance(x_multivariate, y_multivariate, window=0.1), 0.2
    )
    cost_matrix = edr_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = edr_pairwise_distance(x)
    single_to_multiple_distance = edr_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = edr_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        edr_distance,
    )


def test_erp_distance():
    np.testing.assert_almost_equal(
        erp_distance(x_multivariate, y_multivariate), 4408.25
    )
    np.testing.assert_almost_equal(
        erp_distance(x_multivariate, y_multivariate, window=0.1), 1473.0
    )
    cost_matrix = erp_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = erp_pairwise_distance(x)
    single_to_multiple_distance = erp_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = erp_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        erp_distance,
    )


def test_lcss_distance():
    np.testing.assert_almost_equal(lcss_distance(x_multivariate, y_multivariate), 0.8)
    np.testing.assert_almost_equal(
        lcss_distance(x_multivariate, y_multivariate, window=0.1), 0.8
    )
    cost_matrix = lcss_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = lcss_pairwise_distance(x)
    single_to_multiple_distance = lcss_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = lcss_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        lcss_distance,
    )


def test_msm_distance():
    np.testing.assert_almost_equal(msm_distance(x_multivariate, y_multivariate), 164.5)
    np.testing.assert_almost_equal(
        msm_distance(x_multivariate, y_multivariate, window=0.1), 5.0
    )
    np.testing.assert_almost_equal(
        msm_distance(x_multivariate, y_multivariate, independent=False), 164.0
    )
    np.testing.assert_almost_equal(
        msm_distance(x_multivariate, y_multivariate, independent=False, window=0.1), 3.0
    )
    cost_matrix = msm_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    cost_matrix = msm_cost_matrix(
        x_multivariate, y_multivariate, window=0.2, independent=False
    )
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = msm_pairwise_distance(x)
    single_to_multiple_distance = msm_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = msm_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        msm_distance,
    )

    def dependent_msm(x, y):
        return msm_distance(x, y, independent=False)

    pairwise_distance = msm_pairwise_distance(x, independent=False)
    single_to_multiple_distance = msm_from_single_to_multiple_distance(
        x[0], y, independent=False
    )
    multiple_to_multiple_distance = msm_from_multiple_to_multiple_distance(
        x, x, independent=False
    )
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        dependent_msm,
    )


def test_twe_distance():
    np.testing.assert_almost_equal(twe_distance(x_multivariate, y_multivariate), 8815.5)
    np.testing.assert_almost_equal(
        twe_distance(x_multivariate, y_multivariate, window=0.1), 2008.252
    )
    cost_matrix = twe_cost_matrix(x_multivariate, y_multivariate, window=0.2)
    bounding_matrix = create_bounding_matrix(
        x_multivariate.shape[1], y_multivariate.shape[1], window=0.2
    )
    cost_matrix_matches_bounding_matrix(
        cost_matrix, (10, 10), bounding_matrix, out_of_bounds_value=0.0
    )
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = twe_pairwise_distance(x)
    single_to_multiple_distance = twe_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = twe_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x,
        y,
        pairwise_distance,
        single_to_multiple_distance,
        multiple_to_multiple_distance,
        twe_distance,
    )
