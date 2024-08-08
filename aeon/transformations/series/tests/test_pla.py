"""Tests for PLA Series Transformer."""

import numpy as np
import pytest

from aeon.transformations.series._pla import PLASeriesTransformer


@pytest.fixture
def X():
    """Test data."""
    return np.array(
        [
            573.0,
            375.0,
            301.0,
            212.0,
            55.0,
            34.0,
            25.0,
            33.0,
            113.0,
            143.0,
            303.0,
            615.0,
            1226.0,
            1281.0,
            1221.0,
            1081.0,
            866.0,
            1096.0,
            1039.0,
            975.0,
            746.0,
            581.0,
            409.0,
            182.0,
        ]
    )


def test_piecewise_linear_approximation_sliding_window(X):
    """Test PLA transformer."""
    pla_string = PLASeriesTransformer(max_error=100_000, transformer="sliding window")
    result_string = pla_string.fit_transform(X)
    expected = np.array(
        [
            430.95555556,
            371.02222222,
            311.08888889,
            251.15555556,
            191.22222222,
            131.28888889,
            71.35555556,
            11.42222222,
            -48.51111111,
            73.8,
            393.7,
            713.6,
            1033.5,
            1353.4,
            1195.83333333,
            1125.77380952,
            1055.71428571,
            985.6547619,
            915.5952381,
            845.53571429,
            775.47619048,
            705.41666667,
            409.0,
            182.0,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)


def test_piecewise_linear_approximation_top_down(X):
    """Test PLA transformer."""
    pla_string = PLASeriesTransformer(100_000, "top down")
    result_string = pla_string.fit_transform(X)
    expected = np.array(
        [
            500.57142857,
            408.71428571,
            316.85714286,
            225.0,
            133.14285714,
            41.28571429,
            -50.57142857,
            -29.4,
            106.0,
            241.4,
            376.8,
            512.2,
            1319.0,
            1227.0,
            1135.0,
            1043.0,
            951.0,
            1189.28571429,
            1032.28571429,
            875.28571429,
            718.28571429,
            561.28571429,
            404.28571429,
            247.28571429,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)


def test_piecewise_linear_approximation_bottom_up(X):
    """Test PLA transformer."""
    pla_string = PLASeriesTransformer(max_error=100_000, transformer="bottom up")
    result_string = pla_string.fit_transform(X)
    expected = np.array(
        [
            471.33333333,
            394.0952381,
            316.85714286,
            239.61904762,
            162.38095238,
            85.14285714,
            7.9047619,
            -69.33333333,
            43.6,
            210.2,
            376.8,
            543.4,
            1246.33333333,
            1203.98809524,
            1161.64285714,
            1119.29761905,
            1076.95238095,
            1034.60714286,
            992.26190476,
            949.91666667,
            759.1,
            572.7,
            386.3,
            199.9,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)


def test_piecewise_linear_approximation_SWAB(X):
    """Test PLA transformer."""
    pla_string = PLASeriesTransformer(
        max_error=100_000, transformer="SWAB", buffer_size=2
    )
    result_string = pla_string.fit_transform(X)
    expected = np.array(
        [
            538.8,
            423.1,
            307.4,
            191.7,
            48.0,
            40.5,
            33.0,
            25.5,
            43.6,
            210.2,
            376.8,
            543.4,
            1276.5,
            1227.0,
            1177.5,
            1128.0,
            953.5,
            980.5,
            1007.5,
            1034.5,
            759.1,
            572.7,
            386.3,
            199.9,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)

    pla_string = PLASeriesTransformer(
        max_error=100_000, transformer="SWAB", buffer_size=5
    )
    result_string = pla_string.fit_transform(X)
    expected = np.array(
        [
            538.8,
            423.1,
            307.4,
            191.7,
            -72.33333333,
            -4.48809524,
            63.35714286,
            131.20238095,
            199.04761905,
            266.89285714,
            334.73809524,
            402.58333333,
            1335.95454545,
            1260.05454545,
            1184.15454545,
            1108.25454545,
            1032.35454545,
            956.45454545,
            880.55454545,
            804.65454545,
            728.75454545,
            652.85454545,
            576.95454545,
            182.0,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)


def test_piecewise_linear_approximation_check_diff_in_params(X):
    """Test PLA transformer."""
    transformers = ["sliding window", "top down", "bottom up", "swab"]
    for i in range(len(transformers)):
        low_error_pla = PLASeriesTransformer(1, transformers[i])
        high_error_pla = PLASeriesTransformer(float("inf"), transformers[i])
        low_error_result = low_error_pla.fit_transform(X)
        high_error_result = high_error_pla.fit_transform(X)
        assert not np.allclose(low_error_result, high_error_result)


def test_piecewise_linear_approximation_wrong_parameters(X):
    """Test PLA transformer."""
    with pytest.raises(ValueError):
        PLASeriesTransformer(100, "Fake Transformer error").fit_transform(X)
    with pytest.raises(ValueError):
        PLASeriesTransformer("max_error").fit_transform(X)
    with pytest.raises(ValueError):
        PLASeriesTransformer(100, "swab", "buffer_size").fit_transform(X)


def test_piecewise_linear_approximation_one_segment(X):
    """Test PLA transformer."""
    X = X[:2]
    pla = PLASeriesTransformer(10, "bottom up")
    result = pla.fit_transform(X)
    np.testing.assert_array_almost_equal(X, result, decimal=1)
