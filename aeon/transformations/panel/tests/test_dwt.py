# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

from aeon.transformations.panel.dwt import DWTTransformer


# Check that exception is raised for bad num levels.
# input types - string, float, negative int, negative float, empty dict.
# correct input is meant to be a positive integer of 0 or more.
@pytest.mark.parametrize("bad_num_levels", ["str", 1.2, -1.2, -1, {}])
def test_bad_input_args(bad_num_levels):
    X = np.ones(shape=(10, 1, 10))

    if not isinstance(bad_num_levels, int):
        with pytest.raises(TypeError):
            DWTTransformer(n_levels=bad_num_levels).fit_transform(X)
    else:
        with pytest.raises(ValueError):
            DWTTransformer(n_levels=bad_num_levels).fit(X).transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]])

    d = DWTTransformer(n_levels=2).fit(X)
    res = d.transform(X)
    orig = np.array([[[16, 12, -6, 2, -math.sqrt(2), -math.sqrt(2), math.sqrt(2), 0]]])
    np.testing.assert_array_almost_equal(res, orig)
    X = np.array([[[-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3]]])
    d = d.fit(X)
    res = d.transform(X)
    orig = np.array(
        [
            [
                [
                    0.75000,
                    13.25000,
                    -3.25000,
                    -4.75000,
                    -5.303301,
                    -1.414214,
                    8.131728,
                    -4.242641,
                ]
            ]
        ]
    )
    np.testing.assert_array_almost_equal(res, orig)


# This is to test that if num_levels = 0 then no change occurs.
def test_no_levels_does_no_change():
    X = np.array([[[1, 2, 3, 4, 5, 56]]])
    d = DWTTransformer(n_levels=0).fit(X)
    res = d.transform(X)
    np.testing.assert_array_almost_equal(res, X)


@pytest.mark.parametrize("num_levels,corr_series_length", [(2, 12), (3, 11), (4, 12)])
def test_output_dimensions(num_levels, corr_series_length):
    X = np.ones(shape=(10, 1, 13))
    d = DWTTransformer(n_levels=num_levels).fit(X)
    res = d.transform(X)

    n_cases, n_channels, series_length = res.shape

    assert series_length == corr_series_length
    assert n_cases == 10
    assert n_channels == 1


# This is to check that DWT produces the same result along each dimension
def test_dwt_performs_correcly_along_each_dim():
    X = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    d = DWTTransformer(n_levels=3).fit(X)
    res = d.transform(X)
    orig = np.array(
        [
            [
                [
                    9 * math.sqrt(2),
                    -4 * math.sqrt(2),
                    -2,
                    -2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                ],
                [
                    9 * math.sqrt(2),
                    -4 * math.sqrt(2),
                    -2,
                    -2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                    -math.sqrt(2) / 2,
                ],
            ]
        ]
    )
    np.testing.assert_array_almost_equal(res, orig)
