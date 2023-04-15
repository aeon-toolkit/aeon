import numpy as np
from aeon.distance_rework import (
    dtw_distance, dtw_alignment_path, ddtw_distance, ddtw_alignment_path,
    edr_distance, edr_alignment_path, erp_distance, erp_alignment_path,
    lcss_distance, lcss_alignment_path, msm_distance, msm_alignment_path,
    twe_distance, twe_alignment_path, wdtw_distance, wdtw_alignment_path,
    wddtw_distance, wddtw_alignment_path
)


def _test_alignment(distance_alignment_func, expected_result, dist_func):
    x = np.array([[1, 2, 3]])
    y = np.array([[1., 2., 2., 3.]])
    path, dist = distance_alignment_func(x, y)
    np.testing.assert_almost_equal(path, expected_result)
    np.testing.assert_almost_equal(dist, dist_func(x, y))


def test_dtw_alignment():
    _test_alignment(dtw_alignment_path, [(0, 0), (1, 1), (1, 2), (2, 3)], dtw_distance)


def test_ddtw_alignment():
    _test_alignment(ddtw_alignment_path, [(0, 0), (0, 1)], ddtw_distance)


def test_edr_alignment():
    _test_alignment(edr_alignment_path, [(0, 0), (1, 1), (1, 2), (2, 3)], edr_distance)


def test_erp_alignment():
    _test_alignment(erp_alignment_path, [(0, 0), (1, 1), (2, 2), (2, 3)], erp_distance)


def test_lcss_alignment():
    _test_alignment(lcss_alignment_path, [(0, 1), (1, 2), (2, 3)], lcss_distance)


def test_msm_alignment():
    _test_alignment(msm_alignment_path, [(0, 0), (1, 1), (1, 2), (2, 3)], msm_distance)


def test_twe_alignment():
    _test_alignment(twe_alignment_path, [(0, 0), (1, 1), (2, 2), (2, 3)], twe_distance)

def test_wdtw_alignment():
    _test_alignment(wdtw_alignment_path, [(0, 0), (1, 1), (1, 2), (2, 3)], wdtw_distance)

def test_wddtw_alignment():
    _test_alignment(wddtw_alignment_path, [(0, 0), (0, 1)], wddtw_distance)