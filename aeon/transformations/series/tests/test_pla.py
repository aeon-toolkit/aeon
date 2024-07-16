"""Test PLA series transformer."""

import numpy as np
import pytest

from aeon.transformations.series._pla import PiecewiseLinearApproximation


@pytest.fixture
def X():
    """Test values."""
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
    """Test PLA series transformer sliding window returns correctly."""
    pla = PiecewiseLinearApproximation(
        PiecewiseLinearApproximation.Algorithm.SlidingWindow, 100
    )
    result = pla.fit_transform(X)
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
    np.testing.assert_array_almost_equal(result, expected)


def test_piecewise_linear_approximation_top_down(X):
    """Test PLA series transformer top down returns correctly."""
    pla = PiecewiseLinearApproximation(
        PiecewiseLinearApproximation.Algorithm.TopDown, 100
    )
    result = pla.fit_transform(X)
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
    np.testing.assert_array_almost_equal(result, expected)


def test_piecewise_linear_approximation_bottom_up(X):
    """Test PLA series transformer bottom up returns correctly."""
    result = PiecewiseLinearApproximation(
        PiecewiseLinearApproximation.Algorithm.BottomUp, 5
    ).fit_transform(X)
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
    np.testing.assert_array_almost_equal(result, expected)


def test_piecewise_linear_approximation_SWAB(X):
    """Test PLA series transformer SWAB returns correctly."""
    result = PiecewiseLinearApproximation(
        PiecewiseLinearApproximation.Algorithm.SWAB, 5
    ).fit_transform(X)
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
    np.testing.assert_array_almost_equal(result, expected)


def test_piecewise_linear_approximation_check_diff_in_params(X):
    """Test PLA series transformer difference in parameters."""
    transformers = [
        PiecewiseLinearApproximation.Algorithm.SlidingWindow,
        PiecewiseLinearApproximation.Algorithm.TopDown,
        PiecewiseLinearApproximation.Algorithm.BottomUp,
        PiecewiseLinearApproximation.Algorithm.SWAB,
    ]
    for i in range(len(transformers)):
        low_error_pla = PiecewiseLinearApproximation(transformers[i], 1)
        high_error_pla = PiecewiseLinearApproximation(transformers[i], float("inf"))
        low_error_result = low_error_pla.fit_transform(X)
        high_error_result = high_error_pla.fit_transform(X)
        assert not np.allclose(low_error_result, high_error_result)


def test_piecewise_linear_approximation_wrong_parameters(X):
    """Test PLA series transformer errors."""
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation("Fake Transformer", 100)
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(
            PiecewiseLinearApproximation.Algorithm.SWAB, "max_error"
        )
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(
            PiecewiseLinearApproximation.Algorithm.SWAB, 100, "buffer_size"
        )


def test_piecewise_linear_approximation_one_segment(X):
    """Test PLA series transformer on one segment."""
    X = X[:2]
    pla = PiecewiseLinearApproximation(
        PiecewiseLinearApproximation.Algorithm.BottomUp, 10
    )
    result = pla.fit_transform(X)
    assert 0 == len(pla.segment_dense)
    np.testing.assert_array_almost_equal(X, result, decimal=1)
