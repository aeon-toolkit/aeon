"""Tests for numba utils functions related to wavelet transformations."""

__maintainer__ = []

import numpy as np
from numpy.testing import assert_array_almost_equal

from aeon.utils.numba.wavelets import haar_transform, multilevel_haar_transform

# Test data was generated using pywt library (PyWavelets) using the following code:
#     import numpy as np
#     import pywt as wt
#
#     x = np.random.default_rng(42).random(16)
#     print(x)
#     pywt_res = wt.dwt(x, "haar", mode="periodic")
#     print(pywt_res[0])
#     print(pywt_res[1])
#     pywt_res2 = wt.dwt(pywt_res[0], "haar", mode="periodic")
#     print(pywt_res2[0])
#     print(pywt_res2[1])
x = np.array(
    [
        0.77395605,
        0.43887844,
        0.85859792,
        0.69736803,
        0.09417735,
        0.97562235,
        0.7611397,
        0.78606431,
        0.12811363,
        0.45038594,
        0.37079802,
        0.92676499,
        0.64386512,
        0.82276161,
        0.4434142,
        0.22723872,
    ]
)
expected_approx = [
    x,
    np.array(
        [
            0.85760349,
            1.10023407,
            0.75646262,
            1.09403845,
            0.40906097,
            0.91751561,
            1.03706171,
            0.47422323,
        ]
    ),
    np.array([1.38440022, 1.30850185, 0.93803129, 1.06863983]),
]
expected_detail = [
    np.array(
        [
            0.23693565,
            0.11400675,
            -0.62327574,
            -0.01762436,
            -0.22788093,
            -0.39312801,
            -0.12649892,
            0.15285915,
        ]
    ),
    np.array([-0.17156573, -0.23870215, -0.35953172, 0.39798691]),
]


def test_haar_transform():
    """Test the Haar wavelet transform."""
    approx, detail = haar_transform(x)
    assert_array_almost_equal(approx, expected_approx[1])
    assert_array_almost_equal(detail, expected_detail[0])


def test_multilevel_haar_transform():
    """Test the multilevel Haar wavelet transform."""
    approx, detail = multilevel_haar_transform(x, levels=2)
    assert_array_almost_equal(approx[0], expected_approx[0])
    for i in range(2):
        assert_array_almost_equal(approx[i + 1], expected_approx[i + 1])
        assert_array_almost_equal(detail[i], expected_detail[i])
