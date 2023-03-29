# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

from aeon.transformations.panel.slope import SlopeTransformer


# Check that exception is raised for bad num levels.
# input types - string, float, negative int, negative float, empty dict.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_input_args(bad_num_intervals):
    X = np.ones(shape=(10, 1, 10))
    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            SlopeTransformer(n_intervals=bad_num_intervals).fit_transform(X)
    else:
        with pytest.raises(ValueError):
            SlopeTransformer(n_intervals=bad_num_intervals).fit_transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]])

    s = SlopeTransformer(n_intervals=2).fit(X)
    res = s.transform(X)
    orig = np.array([[[(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10]]])
    np.testing.assert_array_almost_equal(res, orig, decimal=2)
    X = np.array([[[-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2]]])
    res = s.fit_transform(X)
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
    np.testing.assert_array_almost_equal(res, orig, decimal=2)


@pytest.mark.parametrize("num_intervals,corr_series_length", [(2, 2), (5, 5), (8, 8)])
def test_output_dimensions(num_intervals, corr_series_length):
    X = np.ones(shape=(10, 1, 10))
    s = SlopeTransformer(n_intervals=num_intervals)
    res = s.fit_transform(X)

    # get the dimension of the generated dataframe.
    num_rows = res.shape[0]
    num_cols = res.shape[1]
    act_time_series_length = res.shape[2]

    assert act_time_series_length == corr_series_length
    assert num_rows == 10
    assert num_cols == 1


# This is to check that Slope produces the same result along each dimension
def test_slope_performs_correcly_along_each_dim():
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]], [[4, 6, 10, 12, 8, 6, 5, 5]]])

    s = SlopeTransformer(n_intervals=2).fit(X)
    res = s.transform(X)
    orig = np.array(
        [
            [[(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10]],
            [[(5 + math.sqrt(41)) / 4, (1 + math.sqrt(101)) / -10]],
        ]
    )
    np.testing.assert_array_almost_equal(res, orig, decimal=2)
