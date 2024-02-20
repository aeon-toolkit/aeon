"""Catch22 test code."""

import numpy as np
import pytest
from numpy import testing

from aeon.datasets import load_basic_motions
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.validation._dependencies import _check_soft_dependencies


def test_catch22_on_basic_motions():
    """Test of Catch22 on basic motions data."""
    # the test currently fails when numba is disabled. See issue #622
    import os

    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        return None

    # load basic motions data
    X_train, _ = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(X_train), 5, replace=False)

    # fit Catch22 and assert transformed data is the same
    c22 = Catch22(replace_nans=True)
    data = c22.fit_transform(X_train[indices])
    testing.assert_array_almost_equal(data, catch22_basic_motions_data, decimal=4)

    # fit Catch22 with select features and assert transformed data is the same
    c22 = Catch22(replace_nans=True, features=feature_names)
    data = c22.fit_transform(X_train[indices])
    testing.assert_array_almost_equal(
        data,
        catch22_basic_motions_data[:, np.sort(np.r_[0:132:22, 5:132:22, 9:132:22])],
        decimal=4,
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("pycatch22", severity="none"),
    reason="skip test if required soft dependency pycatch22 not available",
)
def test_catch22_wrapper_on_basic_motions():
    """Test of Catch22Wrapper on basic motions data."""
    # load basic motions data
    X_train, _ = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(X_train), 5, replace=False)

    # fit Catch22Wrapper and assert transformed data is the same
    c22 = Catch22(use_pycatch22=True, replace_nans=True)
    data = c22.fit_transform(X_train[indices])
    testing.assert_array_almost_equal(
        data, catch22wrapper_basic_motions_data, decimal=4
    )

    # fit Catch22Wrapper with select features and assert transformed data is the same
    c22 = Catch22(use_pycatch22=True, replace_nans=True, features=feature_names)
    data = c22.fit_transform(X_train[indices])
    testing.assert_array_almost_equal(
        data,
        catch22wrapper_basic_motions_data[
            :, np.sort(np.r_[0:132:22, 5:132:22, 9:132:22])
        ],
        decimal=4,
    )


feature_names = ["DN_HistogramMode_5", "CO_f1ecac", "FC_LocalSimple_mean3_stderr"]

catch22_basic_motions_data = np.array(
    [
        [
            0.8918,
            1.2091,
            7.0,
            -0.2,
            0.04,
            2.0,
            3.0,
            0.4158,
            0.9327,
            1.3956,
            0.9846,
            0.0956,
            1.0,
            0.9192,
            4.0,
            2.0822,
            0.5,
            0.0616,
            0.2,
            0.3714,
            0.0108,
            13.0,
            -1.8851,
            -3.4695,
            8.0,
            -0.56,
            0.02,
            3.0,
            6.0,
            10.6256,
            0.4909,
            3.4816,
            -0.9328,
            0.2865,
            2.0,
            0.9192,
            8.0,
            1.8188,
            0.75,
            0.0934,
            0.1714,
            0.2857,
            0.0168,
            11.0,
            0.065,
            0.2902,
            9.0,
            -0.78,
            -0.34,
            2.0,
            5.0,
            0.3406,
            0.6381,
            0.914,
            0.1154,
            0.1343,
            2.0,
            0.8485,
            8.0,
            1.9982,
            0.3333,
            0.0967,
            0.6857,
            0.3143,
            0.0034,
            10.0,
            0.0216,
            0.0216,
            10.0,
            -0.38,
            0.06,
            2.0,
            7.0,
            0.1938,
            0.6381,
            0.7083,
            -0.0848,
            0.1159,
            2.0,
            0.899,
            5.0,
            2.0236,
            0.3333,
            0.1549,
            0.1714,
            0.3143,
            0.0052,
            12.0,
            0.3028,
            0.1962,
            10.0,
            0.03,
            0.35,
            2.0,
            7.0,
            0.1221,
            0.4909,
            0.4555,
            0.0199,
            0.1564,
            2.0,
            0.8283,
            8.0,
            1.9428,
            0.6667,
            0.5965,
            0.8,
            0.2286,
            0.0037,
            13.0,
            0.5644,
            0.9089,
            8.0,
            -0.11,
            -0.72,
            3.0,
            6.0,
            1.7181,
            0.4909,
            1.4064,
            -0.4162,
            0.1914,
            2.0,
            0.9091,
            8.0,
            1.8064,
            1.0,
            0.1072,
            0.7429,
            0.2,
            0.0125,
            11.0,
        ],
        [
            0.1467,
            1.7614,
            9.0,
            -0.15,
            0.03,
            2.0,
            8.0,
            32.285,
            0.4909,
            7.0546,
            159.5298,
            0.1971,
            4.0,
            0.9091,
            8.0,
            1.9656,
            0.3333,
            0.1012,
            0.1714,
            0.6571,
            0.0091,
            3.0,
            -1.031,
            0.5692,
            13.0,
            0.2,
            -0.37,
            2.0,
            3.0,
            18.3029,
            0.7363,
            7.4032,
            -298.4341,
            0.1008,
            3.0,
            0.9091,
            7.0,
            2.0451,
            0.3333,
            0.0992,
            0.3143,
            0.2,
            0.0015,
            8.0,
            0.1721,
            -1.3777,
            12.0,
            -0.23,
            -0.04,
            2.0,
            3.0,
            14.3938,
            0.7854,
            6.47,
            91.0107,
            0.156,
            3.0,
            0.9091,
            5.0,
            2.0199,
            0.3333,
            0.1303,
            0.7429,
            0.5429,
            0.0107,
            13.0,
            -0.2157,
            -1.7152,
            13.0,
            0.15,
            -0.18,
            2.0,
            5.0,
            7.1407,
            0.6872,
            4.2252,
            4.1153,
            0.0992,
            2.0,
            0.899,
            6.0,
            1.9989,
            0.3333,
            0.1047,
            0.6571,
            0.2286,
            0.0024,
            14.0,
            0.6856,
            -0.1177,
            14.0,
            0.18,
            0.3,
            2.0,
            4.0,
            1.6007,
            0.8345,
            2.9957,
            3.2192,
            0.0709,
            1.0,
            0.8687,
            5.0,
            2.0294,
            0.3333,
            0.064,
            0.7429,
            0.8286,
            0.0061,
            6.0,
            -1.5437,
            0.0972,
            8.0,
            -0.14,
            0.1,
            2.0,
            5.0,
            7.3076,
            0.6872,
            5.1671,
            119.5049,
            0.0292,
            2.0,
            0.8889,
            5.0,
            1.985,
            0.3333,
            0.1307,
            0.2286,
            0.2,
            0.007,
            8.0,
        ],
        [
            -0.2176,
            -0.252,
            7.0,
            -0.13,
            0.02,
            1.0,
            6.0,
            0.0081,
            0.8836,
            0.1506,
            -0.0018,
            0.0569,
            2.0,
            0.697,
            5.0,
            2.1494,
            0.3333,
            1.6321,
            0.6,
            0.2,
            0.0024,
            19.0,
            0.2537,
            0.3414,
            8.0,
            -0.12,
            -0.02,
            2.0,
            5.0,
            0.1288,
            0.589,
            0.4545,
            0.0234,
            0.213,
            2.0,
            0.8586,
            7.0,
            1.9398,
            1.0,
            0.5715,
            0.8286,
            0.2,
            0.0107,
            10.0,
            -0.045,
            -0.0049,
            9.0,
            -0.91,
            -0.22,
            2.0,
            3.0,
            0.0037,
            0.9327,
            0.1481,
            -0.0013,
            0.0428,
            1.0,
            0.6162,
            6.0,
            2.0848,
            1.0,
            1.2419,
            0.6,
            0.1714,
            0.0004,
            0.0,
            -0.0368,
            -0.0176,
            11.0,
            -0.39,
            -0.08,
            2.0,
            5.0,
            0.0041,
            0.589,
            0.077,
            -0.0,
            0.1782,
            2.0,
            0.3939,
            7.0,
            1.9786,
            0.25,
            3.976,
            0.7429,
            0.2571,
            0.0295,
            0.0,
            0.02,
            0.0044,
            8.0,
            -0.06,
            0.05,
            2.0,
            5.0,
            0.0013,
            0.589,
            0.0467,
            -0.0,
            0.0615,
            2.0,
            0.2424,
            7.0,
            2.0341,
            0.3333,
            6.8497,
            0.7429,
            0.2,
            0.0034,
            0.0,
            -0.0914,
            -0.1258,
            11.0,
            -0.155,
            0.125,
            2.0,
            5.0,
            0.0212,
            0.589,
            0.1719,
            0.0001,
            0.1025,
            2.0,
            0.7071,
            7.0,
            1.8809,
            1.0,
            3.1416,
            0.8,
            0.2,
            0.0024,
            10.0,
        ],
        [
            14.753,
            12.6115,
            8.0,
            -0.01,
            -0.17,
            2.0,
            5.0,
            8.3186,
            0.7363,
            15.168,
            611.2311,
            0.2837,
            1.0,
            0.8485,
            5.0,
            2.058,
            1.0,
            0.1723,
            0.2571,
            0.2286,
            0.0046,
            7.0,
            -8.7478,
            -11.0416,
            5.0,
            0.09,
            0.01,
            2.0,
            4.0,
            5.3016,
            0.7363,
            16.0359,
            -666.9228,
            0.2358,
            1.0,
            0.8586,
            4.0,
            2.0048,
            1.0,
            0.1222,
            0.8286,
            0.1714,
            0.005,
            7.0,
            -1.1495,
            -3.2478,
            8.0,
            0.13,
            -0.08,
            1.0,
            2.0,
            1.7627,
            1.4726,
            3.327,
            3.9326,
            0.095,
            2.0,
            0.8586,
            5.0,
            2.1462,
            0.5,
            0.0841,
            0.6,
            0.7714,
            0.0012,
            3.0,
            0.0946,
            -1.7292,
            5.0,
            0.06,
            0.12,
            1.0,
            3.0,
            0.5924,
            0.8836,
            2.8071,
            13.9244,
            0.1318,
            1.0,
            0.8586,
            7.0,
            2.1384,
            0.5,
            0.0608,
            0.8286,
            0.2286,
            0.0124,
            8.0,
            -0.2413,
            -0.7284,
            6.0,
            -0.05,
            -0.11,
            2.0,
            4.0,
            0.2086,
            0.7363,
            2.8863,
            -0.314,
            0.1682,
            1.0,
            0.8384,
            6.0,
            2.0693,
            0.5,
            0.0498,
            0.8286,
            0.1714,
            0.0165,
            8.0,
            -0.2211,
            0.9037,
            6.0,
            0.01,
            0.04,
            2.0,
            4.0,
            1.3957,
            0.7363,
            7.0085,
            -63.8967,
            0.2979,
            1.0,
            0.8586,
            5.0,
            1.9304,
            0.6667,
            0.0848,
            0.8286,
            0.1714,
            0.0162,
            8.0,
        ],
        [
            -0.0619,
            0.1991,
            6.0,
            0.03,
            0.13,
            2.0,
            4.0,
            0.501,
            0.8836,
            1.5209,
            1.2447,
            0.1921,
            1.0,
            0.899,
            6.0,
            2.0481,
            0.3333,
            0.0718,
            0.1714,
            0.2,
            0.0092,
            6.0,
            -3.0176,
            -3.5235,
            9.0,
            0.0,
            0.1,
            3.0,
            7.0,
            7.6385,
            0.4418,
            2.4961,
            -1.2227,
            0.3019,
            2.0,
            0.9091,
            8.0,
            1.7771,
            1.0,
            0.099,
            0.1714,
            0.3143,
            0.0081,
            13.0,
            -0.519,
            -0.6649,
            14.0,
            -0.13,
            0.19,
            3.0,
            6.0,
            0.3096,
            0.4418,
            0.6655,
            -0.0086,
            0.2148,
            2.0,
            0.8384,
            6.0,
            1.8777,
            0.5,
            0.4267,
            0.6571,
            0.2,
            0.0133,
            13.0,
            -0.2996,
            -0.0955,
            8.0,
            0.18,
            0.22,
            3.0,
            8.0,
            0.379,
            0.4418,
            0.7076,
            -0.1388,
            0.2115,
            2.0,
            0.8384,
            7.0,
            1.9195,
            0.5,
            0.2553,
            0.1714,
            0.2857,
            0.0064,
            14.0,
            -0.3873,
            0.4293,
            9.0,
            0.11,
            0.35,
            3.0,
            7.0,
            0.2719,
            0.4418,
            0.4774,
            0.0065,
            0.2377,
            3.0,
            0.8081,
            7.0,
            1.7775,
            1.0,
            0.6829,
            0.1714,
            0.2571,
            0.0098,
            13.0,
            -0.9868,
            -1.4515,
            9.0,
            -0.04,
            0.08,
            3.0,
            7.0,
            1.5658,
            0.4418,
            1.1244,
            -0.0475,
            0.2511,
            3.0,
            0.8687,
            8.0,
            1.7647,
            1.0,
            0.343,
            0.1714,
            0.2857,
            0.0168,
            13.0,
        ],
    ]
)

catch22wrapper_basic_motions_data = np.array(
    [
        [
            0.0804,
            0.3578,
            4.0,
            -0.28,
            0.04,
            1.1428,
            3.0,
            0.3176,
            0.9327,
            1.226,
            0.6572,
            0.0956,
            1.0,
            0.9192,
            7.0,
            2.0909,
            0.5,
            0.0653,
            0.2,
            0.3714,
            0.0108,
            13.0,
            -0.5321,
            -0.9703,
            8.0,
            -0.05,
            0.02,
            2.2361,
            6.0,
            0.8128,
            0.4909,
            0.968,
            -0.0197,
            0.281,
            2.0,
            0.8889,
            8.0,
            1.8244,
            0.75,
            0.3456,
            0.1714,
            0.2857,
            0.0243,
            11.0,
            0.2931,
            0.5641,
            8.0,
            -0.78,
            -0.34,
            1.5664,
            5.0,
            0.4932,
            0.6381,
            1.1057,
            0.2011,
            0.1343,
            2.0,
            0.8485,
            9.0,
            1.9998,
            0.3333,
            0.0597,
            0.6857,
            0.3143,
            0.0046,
            10.0,
            0.0448,
            0.0448,
            5.0,
            -0.38,
            0.06,
            1.2329,
            7.0,
            0.474,
            0.6381,
            1.1134,
            -0.3243,
            0.1159,
            2.0,
            0.899,
            10.0,
            2.0078,
            0.3333,
            0.0806,
            0.1714,
            0.3143,
            0.0101,
            12.0,
            0.7638,
            0.5217,
            8.0,
            0.03,
            0.35,
            1.815,
            7.0,
            0.6291,
            0.4909,
            1.0392,
            0.2326,
            0.13,
            2.0,
            0.899,
            10.0,
            1.9443,
            0.6667,
            0.1696,
            0.8,
            0.2286,
            0.0046,
            13.0,
            0.4149,
            0.6463,
            8.0,
            -0.11,
            -0.805,
            2.1736,
            6.0,
            0.7752,
            0.4909,
            0.9495,
            -0.1261,
            0.185,
            2.0,
            0.9091,
            8.0,
            1.8142,
            1.0,
            0.2083,
            0.7429,
            0.2,
            0.0174,
            11.0,
        ],
        [
            -0.8062,
            -0.5798,
            8.0,
            -0.16,
            0.01,
            1.851,
            8.0,
            0.635,
            0.4909,
            0.9945,
            0.44,
            0.1823,
            4.0,
            0.8586,
            9.0,
            1.9501,
            0.3333,
            0.1544,
            0.1714,
            0.6571,
            0.0091,
            24.0,
            -0.1266,
            0.1157,
            7.0,
            0.2,
            -0.37,
            1.3705,
            3.0,
            0.4195,
            0.7363,
            1.1266,
            -1.0354,
            0.1041,
            3.0,
            0.8889,
            13.0,
            2.0298,
            0.3333,
            0.0602,
            0.3143,
            0.2,
            0.0037,
            8.0,
            0.4734,
            0.207,
            5.0,
            -0.23,
            0.06,
            1.2531,
            3.0,
            0.4252,
            0.7854,
            1.1178,
            0.4621,
            0.1742,
            3.0,
            0.8687,
            12.0,
            2.0059,
            0.3333,
            0.0523,
            0.7429,
            0.5429,
            0.0116,
            13.0,
            -0.0013,
            -0.3798,
            6.0,
            0.06,
            -0.2,
            1.6441,
            5.0,
            0.4549,
            0.6872,
            1.0719,
            0.0662,
            0.0992,
            2.0,
            0.8788,
            13.0,
            2.0097,
            0.3333,
            0.0879,
            0.6571,
            0.2286,
            0.0037,
            14.0,
            0.2664,
            -0.0514,
            5.0,
            0.19,
            0.3,
            1.2534,
            4.0,
            0.2506,
            0.8345,
            1.1916,
            0.1995,
            0.0692,
            1.0,
            0.8081,
            14.0,
            2.0191,
            0.3333,
            0.0673,
            0.7429,
            0.8286,
            0.0073,
            6.0,
            -0.3504,
            0.0039,
            5.0,
            -0.14,
            -0.04,
            1.1481,
            5.0,
            0.3408,
            0.6872,
            1.1216,
            1.2034,
            0.0276,
            2.0,
            0.7778,
            8.0,
            1.9736,
            0.3333,
            0.0342,
            0.2286,
            0.2,
            0.0092,
            8.0,
        ],
        [
            0.3645,
            0.1094,
            5.0,
            -0.13,
            -0.34,
            0.9342,
            6.0,
            0.4448,
            0.8836,
            1.1237,
            -0.7506,
            0.0879,
            2.0,
            0.8687,
            7.0,
            2.133,
            0.3333,
            0.0845,
            0.6,
            0.2,
            0.0024,
            7.0,
            0.3658,
            0.5744,
            7.0,
            -0.23,
            -0.01,
            1.8247,
            5.0,
            0.7293,
            0.589,
            1.087,
            0.3158,
            0.2149,
            2.0,
            0.8687,
            8.0,
            1.9505,
            1.0,
            0.1758,
            0.8286,
            0.2,
            0.0116,
            10.0,
            -0.3643,
            -0.0421,
            6.0,
            -0.91,
            -0.22,
            1.1595,
            3.0,
            0.2371,
            0.9327,
            1.1946,
            -0.6767,
            0.13,
            1.0,
            0.8182,
            9.0,
            2.0869,
            1.0,
            0.0611,
            0.6,
            0.1714,
            0.0004,
            7.0,
            -0.3788,
            -0.1279,
            7.0,
            -0.39,
            -0.08,
            1.8915,
            5.0,
            0.6975,
            0.589,
            1.0127,
            -0.0644,
            0.1358,
            2.0,
            0.8586,
            11.0,
            1.9728,
            0.25,
            0.1512,
            0.7429,
            0.2571,
            0.0122,
            10.0,
            0.2208,
            -0.1147,
            7.0,
            -0.08,
            0.03,
            1.7819,
            5.0,
            0.6061,
            0.589,
            1.0105,
            -0.1311,
            0.0862,
            2.0,
            0.8687,
            8.0,
            2.039,
            0.3333,
            0.1591,
            0.7429,
            0.2,
            0.0043,
            12.0,
            -0.6362,
            -0.8457,
            7.0,
            -0.2,
            0.12,
            1.9983,
            5.0,
            0.7828,
            0.589,
            1.0502,
            0.0228,
            0.1284,
            2.0,
            0.8586,
            11.0,
            1.8706,
            1.0,
            0.3778,
            0.8,
            0.2,
            0.0037,
            10.0,
        ],
        [
            0.7285,
            0.5526,
            5.0,
            -0.07,
            -0.17,
            1.3139,
            5.0,
            0.0561,
            0.7363,
            1.2518,
            0.3383,
            0.254,
            1.0,
            0.8081,
            8.0,
            2.0538,
            1.0,
            0.1392,
            0.2571,
            0.2286,
            0.0046,
            7.0,
            -0.3475,
            -0.5286,
            4.0,
            0.08,
            -0.01,
            1.2448,
            4.0,
            0.0331,
            0.7363,
            1.2729,
            -0.3284,
            0.2395,
            1.0,
            0.8283,
            5.0,
            2.0075,
            1.0,
            0.1219,
            0.8286,
            0.1714,
            0.005,
            7.0,
            0.4846,
            -0.2633,
            5.0,
            -0.02,
            -0.12,
            0.8858,
            2.0,
            0.2239,
            1.4726,
            1.192,
            0.1781,
            0.1024,
            2.0,
            0.8485,
            8.0,
            2.1476,
            0.5,
            0.0817,
            0.6,
            0.7714,
            0.0012,
            3.0,
            0.0544,
            -0.747,
            7.0,
            0.06,
            0.12,
            0.9238,
            3.0,
            0.1144,
            0.8836,
            1.2399,
            1.1815,
            0.1503,
            1.0,
            0.8283,
            5.0,
            2.1388,
            0.5,
            0.0664,
            0.8286,
            0.2286,
            0.0124,
            8.0,
            -0.1774,
            -0.3877,
            6.0,
            -0.05,
            -0.1,
            1.124,
            4.0,
            0.0388,
            0.7363,
            1.2521,
            -0.0252,
            0.1556,
            1.0,
            0.8182,
            6.0,
            2.0597,
            0.5,
            0.086,
            0.8286,
            0.1714,
            0.0165,
            8.0,
            -0.0394,
            0.1582,
            5.0,
            0.01,
            0.04,
            1.473,
            4.0,
            0.0431,
            0.7363,
            1.2381,
            -0.3468,
            0.3,
            1.0,
            0.8586,
            6.0,
            1.9356,
            0.6667,
            0.2129,
            0.8286,
            0.1714,
            0.0165,
            8.0,
        ],
        [
            -0.6177,
            -0.4164,
            6.0,
            0.03,
            -0.14,
            1.2578,
            4.0,
            0.2982,
            0.8836,
            1.1794,
            0.5714,
            0.1921,
            1.0,
            0.899,
            6.0,
            2.0492,
            0.3333,
            0.1321,
            0.1714,
            0.2,
            0.0089,
            6.0,
            -1.0401,
            -1.2153,
            8.0,
            0.0,
            0.1,
            2.5291,
            7.0,
            0.9158,
            0.4418,
            0.8688,
            -0.0508,
            0.3019,
            2.0,
            0.8889,
            9.0,
            1.7807,
            1.0,
            0.3581,
            0.1714,
            0.3143,
            0.0156,
            13.0,
            -0.2369,
            -0.4506,
            6.0,
            -0.13,
            0.3,
            2.0582,
            6.0,
            0.665,
            0.4418,
            0.9804,
            -0.0271,
            0.2042,
            2.0,
            0.8586,
            14.0,
            1.8881,
            0.5,
            0.2277,
            0.6571,
            0.2,
            0.0122,
            13.0,
            -0.4152,
            -0.1326,
            7.0,
            0.18,
            0.22,
            2.0035,
            8.0,
            0.7259,
            0.4418,
            0.9844,
            -0.3678,
            0.1768,
            2.0,
            0.8384,
            8.0,
            1.9223,
            0.5,
            0.1498,
            0.1714,
            0.2857,
            0.0069,
            14.0,
            -0.7112,
            0.7554,
            7.0,
            0.11,
            0.22,
            2.5648,
            7.0,
            0.8769,
            0.4418,
            0.8619,
            0.0377,
            0.2621,
            3.0,
            0.8283,
            9.0,
            1.7647,
            1.0,
            0.317,
            0.1714,
            0.2571,
            0.0122,
            13.0,
            -0.7754,
            -1.133,
            8.0,
            -0.04,
            0.08,
            2.5809,
            7.0,
            0.9274,
            0.4418,
            0.8698,
            -0.0217,
            0.2648,
            3.0,
            0.8485,
            9.0,
            1.7683,
            1.0,
            0.4885,
            0.1714,
            0.2857,
            0.0208,
            13.0,
        ],
    ]
)
