"""Tests for SlopeTransformer."""

import math

import numpy as np
import pytest

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


def test_output_of_transformer():
    """Check the transformer has changed the data correctly."""
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]])
    s = SlopeTransformer(n_intervals=2).fit(X)
    res = s.transform(X)
    orig = np.array([[[(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10]]])
    np.testing.assert_array_almost_equal(res, orig, decimal=5)

    X = np.array([[[-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2]]])
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
    """Check that Slope produces the same result along each dimension."""
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
