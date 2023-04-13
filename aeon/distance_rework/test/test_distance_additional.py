import numpy as np
from aeon.distance_rework.test._utils import create_test_distance_numpy
from aeon.distance_rework import (
    dtw_pairwise_distance,
    dtw_distance,
    dtw_from_single_to_multiple_distance,
    dtw_from_multiple_to_multiple_distance,
    euclidean_distance,
    euclidean_pairwise_distance,
    euclidean_from_single_to_multiple_distance,
    euclidean_from_multiple_to_multiple_distance,
    squared_distance,
    squared_pairwise_distance,
    squared_from_single_to_multiple_distance,
    squared_from_multiple_to_multiple_distance,
    ddtw_pairwise_distance,
    ddtw_distance,
    ddtw_from_single_to_multiple_distance,
    ddtw_from_multiple_to_multiple_distance,
    wdtw_pairwise_distance,
    wdtw_distance,
    wdtw_from_single_to_multiple_distance,
    wdtw_from_multiple_to_multiple_distance,
    wddtw_pairwise_distance,
    wddtw_distance,
    wddtw_from_single_to_multiple_distance,
    wddtw_from_multiple_to_multiple_distance,
    edr_pairwise_distance,
    edr_distance,
    edr_from_single_to_multiple_distance,
    edr_from_multiple_to_multiple_distance,
    erp_pairwise_distance,
    erp_distance,
    erp_from_single_to_multiple_distance,
    erp_from_multiple_to_multiple_distance,
    lcss_pairwise_distance,
    lcss_distance,
    lcss_from_single_to_multiple_distance,
    lcss_from_multiple_to_multiple_distance,
    twe_pairwise_distance,
    twe_distance,
    twe_from_single_to_multiple_distance,
    twe_from_multiple_to_multiple_distance,
    msm_pairwise_distance,
    msm_distance,
    msm_from_single_to_multiple_distance,
    msm_from_multiple_to_multiple_distance,
)
from aeon.distance_rework.test._utils import almost_equal


def _additional_distance_tests(x, y, pairwise_distance, single_to_multiple_distance,
                               multiple_to_multiple_distance, distance_function):
    assert pairwise_distance.shape == (5, 5)
    assert single_to_multiple_distance.shape == (5,)
    assert multiple_to_multiple_distance.shape == (5, 5)

    for i in range(pairwise_distance.shape[0]):
        for j in range(pairwise_distance.shape[0]):
            almost_equal(pairwise_distance[i, j], multiple_to_multiple_distance[i, j])
            almost_equal(pairwise_distance[i, j], distance_function(x[i], x[j]))

    for i in range(single_to_multiple_distance.shape[0]):
        almost_equal(single_to_multiple_distance[i], distance_function(x[0], y[i]))


def test_euclidean_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = euclidean_pairwise_distance(x)
    single_to_multiple_distance = euclidean_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = euclidean_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, euclidean_distance
    )


def test_squared_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = squared_pairwise_distance(x)
    single_to_multiple_distance = squared_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = squared_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, squared_distance
    )


def test_dtw_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = dtw_pairwise_distance(x)
    single_to_multiple_distance = dtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = dtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, dtw_distance
    )


def test_ddtw_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = ddtw_pairwise_distance(x)
    single_to_multiple_distance = ddtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = ddtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, ddtw_distance
    )


# TODO: Add univariate tests
# TODO: These are now broken so fix them idk why


def test_wdtw_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = wdtw_pairwise_distance(x)
    single_to_multiple_distance = wdtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = wdtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, wdtw_distance
    )


def test_wddtw_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = wddtw_pairwise_distance(x)
    single_to_multiple_distance = wddtw_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = wddtw_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, wddtw_distance
    )


def test_edr_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = edr_pairwise_distance(x)
    single_to_multiple_distance = edr_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = edr_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, edr_distance
    )


def test_erp_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = erp_pairwise_distance(x)
    single_to_multiple_distance = erp_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = erp_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, erp_distance
    )


def test_lcss_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = lcss_pairwise_distance(x)
    single_to_multiple_distance = lcss_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = lcss_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, lcss_distance
    )


def test_msm_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = msm_pairwise_distance(x)
    single_to_multiple_distance = msm_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = msm_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, msm_distance
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
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, dependent_msm
    )


def test_twe_distance():
    x = create_test_distance_numpy(5, 2, 10, 1)
    y = create_test_distance_numpy(5, 2, 10, 2)
    pairwise_distance = twe_pairwise_distance(x)
    single_to_multiple_distance = twe_from_single_to_multiple_distance(x[0], y)
    multiple_to_multiple_distance = twe_from_multiple_to_multiple_distance(x, x)
    _additional_distance_tests(
        x, y, pairwise_distance, single_to_multiple_distance,
        multiple_to_multiple_distance, twe_distance
    )
