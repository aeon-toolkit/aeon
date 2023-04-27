# -*- coding: utf-8 -*-
import numpy as np
import pytest

from aeon.transformations.panel.dictionary_based._paa import PAA


# Check that exception is raised for bad num intervals.
# input types - string, float, negative int, negative float, empty dict
# and an int that is larger than the time series length.
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 11, 0])
def test_bad_input_args(bad_num_intervals):
    X = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            PAA(n_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            PAA(n_intervals=bad_num_intervals).fit_transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():
    X = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    p = PAA(n_intervals=3).fit(X)
    res = p.transform(X)
    orig = np.array([[[2.2, 5.5, 8.8]]])
    np.testing.assert_array_almost_equal(res, orig)


def test_output_dimensions():
    # test with univariate
    X = np.random.rand(10, 1, 12)
    p = PAA(n_intervals=5).fit(X)
    res = p.transform(X)
    assert res.shape[0] == 10 and res.shape[1] == 1 and res.shape[2] == 5


# This is to check that PAA produces the same result along each dimension
def test_paa_performs_correcly_along_each_dim():
    X = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    p = PAA(n_intervals=3).fit(X)
    res = p.transform(X)
    orig = [[[2.2, 5.5, 8.8], [2.2, 5.5, 8.8]]]
    np.testing.assert_array_almost_equal(res, orig)
