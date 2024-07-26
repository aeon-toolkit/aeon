"""Tests for PLA Series Transformer."""

import numpy as np
import pytest

from aeon.transformations.series._pla import PiecewiseLinearApproximation


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
    pla_string = PiecewiseLinearApproximation(100, "sliding window")
    result_string = pla_string.fit_transform(X)
    pla_num = PiecewiseLinearApproximation(100, 1)
    result_num = pla_num.fit_transform(X)
    expected = np.array(
        [
            573.0,
            375.0,
            301.0,
            212.0,
            53.0,
            38.0,
            23.0,
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
            1097.16666667,
            1036.66666667,
            976.16666667,
            747.16666667,
            578.66666667,
            410.16666667,
            182.0,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)
    np.testing.assert_array_almost_equal(result_num, expected)


def test_piecewise_linear_approximation_top_down(X):
    """Test PLA transformer."""
    pla_string = PiecewiseLinearApproximation(100, "top down")
    result_string = pla_string.fit_transform(X)
    pla_num = PiecewiseLinearApproximation(100, 2)
    result_num = pla_num.fit_transform(X)
    expected = np.array(
        [
            573.0,
            375.0,
            301.0,
            212.0,
            53.0,
            38.0,
            23.0,
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
            1097.16666667,
            1036.66666667,
            976.16666667,
            746.0,
            581.0,
            409.0,
            182.0,
        ]
    )
    np.testing.assert_array_almost_equal(result_string, expected)
    np.testing.assert_array_almost_equal(result_num, expected)


def test_piecewise_linear_approximation_bottom_up(X):
    """Test PLA transformer."""
    pla_string = PiecewiseLinearApproximation(5, "bottom up")
    result_string = pla_string.fit_transform(X)
    pla_num = PiecewiseLinearApproximation(5, 3)
    result_num = pla_num.fit_transform(X)
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
    np.testing.assert_array_almost_equal(result_num, expected)


def test_piecewise_linear_approximation_SWAB(X):
    """Test PLA transformer."""
    pla_string = PiecewiseLinearApproximation(5, "swab")
    result_string = pla_string.fit_transform(X)
    pla_num = PiecewiseLinearApproximation(5, 4)
    result_num = pla_num.fit_transform(X)
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
    np.testing.assert_array_almost_equal(result_num, expected)


def test_piecewise_linear_approximation_check_diff_in_params(X):
    """Test PLA transformer."""
    transformers = ["sliding window", "top down", "bottom up", "swab"]
    for i in range(len(transformers)):
        low_error_pla = PiecewiseLinearApproximation(1, transformers[i])
        high_error_pla = PiecewiseLinearApproximation(float("inf"), transformers[i])
        low_error_result = low_error_pla.fit_transform(X)
        high_error_result = high_error_pla.fit_transform(X)
        assert not np.allclose(low_error_result, high_error_result)


def test_piecewise_linear_approximation_wrong_parameters(X):
    """Test PLA transformer."""
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(100, "Fake Transformer error").fit_transform(X)
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(100, 5).fit_transform(X)
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation("max_error").fit_transform(X)
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(100, "swab", "buffer_size").fit_transform(X)


def test_piecewise_linear_approximation_one_segment(X):
    """Test PLA transformer."""
    X = X[:2]
    pla = PiecewiseLinearApproximation(10, "bottom up")
    result = pla.fit_transform(X)
    assert pla.segment_dense is None
    np.testing.assert_array_almost_equal(X, result, decimal=1)
