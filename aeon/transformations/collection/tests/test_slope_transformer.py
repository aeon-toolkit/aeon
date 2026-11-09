"""Tests for SlopeTransformer."""

import math

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import SlopeTransformer


@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_input_args(bad_num_intervals):
    """Check that exception is raised for bad num levels.

    input types - string, float, negative int, negative float, empty dict.
    correct input is meant to be a positive integer of 1 or more.
    """
    X = np.ones(shape=(10, 10, 1))
    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            SlopeTransformer(n_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            SlopeTransformer(n_intervals=bad_num_intervals).fit(X).transform(X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_of_transformer(dtype):
    """Check the transformer has changed the data correctly."""
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]], dtype=dtype)
    s = SlopeTransformer(n_intervals=2).fit(X)
    res = s.transform(X)
    orig = np.array([[[(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10]]])
    np.testing.assert_array_almost_equal(res, orig, decimal=5)

    X = np.array([[[-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2]]], dtype=dtype)
    s = s.fit(X)
    res = s.transform(X)
    orig = np.array(
        [
            [
                [
                    (104.8 + math.sqrt(14704.04)) / 61,
                    (143.752 + math.sqrt(20790.0775)) / -11.2,
                ]
            ]
        ]
    )
    np.testing.assert_array_almost_equal(res, orig, decimal=5)


@pytest.mark.parametrize("num_intervals,corr_n_timepoints", [(2, 2), (5, 5), (8, 8)])
def test_output_dimensions(num_intervals, corr_n_timepoints):
    """Test the output dimensions of SlopeTransformer."""
    X = np.ones(shape=(10, 1, 13))
    s = SlopeTransformer(n_intervals=num_intervals).fit(X)
    res = s.transform(X)
    n_cases, n_channels, n_timepoints = res.shape

    assert n_timepoints == corr_n_timepoints
    assert n_cases == 10
    assert n_channels == 1


def test_slope_performs_correcly_along_each_dim():
    """Check that Slope produces the same result along each channel."""
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5], [4, 6, 10, 12, 8, 6, 5, 5]]])
    s = SlopeTransformer(n_intervals=2).fit(X)
    res = s.transform(X)
    orig = np.array(
        [
            [
                [(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10],
                [(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10],
            ]
        ]
    )
    np.testing.assert_array_almost_equal(res, orig, decimal=5)


@pytest.mark.parametrize(
    "dtype",
    ["int16", "int32", "int64", "float32", "float64"],
)
def test_slope_preserves_float_precision(dtype):
    """Check that Slope preserved float32 and promotes proper dtype output."""
    X = make_example_3d_numpy(
        n_cases=2,
        n_channels=1,
        n_timepoints=8,
        return_y=False,
        random_state=0,
    ).astype(dtype)

    Xt = SlopeTransformer(n_intervals=2).fit_transform(X)

    expected_dtype = np.float32 if dtype == "float32" else np.float64
    assert Xt.dtype == expected_dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("slope", [1e-4, -1e-4, 1.0, -1.0, 1e9, -1e9])
@pytest.mark.parametrize("n_intervals", [1, 2])
def test_slope_linear_values(dtype, slope, n_intervals):
    """Check signed slopes remain accurate without cancellation or overflow."""
    X = (np.arange(1, 9, dtype=dtype) * slope)[None, None, :]

    Xt = SlopeTransformer(n_intervals=n_intervals).fit_transform(X)

    assert Xt.shape == (1, 1, n_intervals)
    assert Xt.dtype == dtype
    assert np.all(np.isfinite(Xt))
    assert np.all(np.sign(Xt) == np.sign(slope))
    # Account for floating-point rounding in the mean and variance.
    rtol = 5e-7 if dtype == np.float32 else 1e-14
    np.testing.assert_allclose(Xt, slope, rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("series", [[3], [3, 3, 3, 3], [1, -1, -1, 1]])
def test_slope_zero_correlation(dtype, series):
    """Preserve zero gradients for singleton, constant and uncorrelated series."""
    X = np.array(series, dtype=dtype)[None, None, :]

    Xt = SlopeTransformer(n_intervals=1).fit_transform(X)

    assert Xt.dtype == dtype
    np.testing.assert_array_equal(Xt, np.zeros((1, 1, 1), dtype=dtype))
