"""Test for HOG1DTransformer."""

import numbers

import numpy as np
import pytest

from aeon.transformations.collection._hog1d import HOG1DTransformer


def test_hog1d():
    """Test HOG1D Transformer exceptions."""
    X = np.random.random((2, 1, 10))
    hog = HOG1DTransformer(n_bins=0)
    with pytest.raises(ValueError, match=r"num_bins must have the value of at least 1"):
        hog.fit_transform(X)
    hog = HOG1DTransformer(n_bins=0.5)
    with pytest.raises(TypeError, match=r"must be an 'int'"):
        hog.fit_transform(X)
    hog = HOG1DTransformer(scaling_factor="Bob")
    with pytest.raises(TypeError, match=r"scaling_factor must be a 'number'"):
        hog.fit_transform(X)


@pytest.mark.parametrize("num_bins,corr_n_timepoints", [(4, 8), (8, 16), (12, 24)])
def test_output_dimensions(num_bins, corr_n_timepoints):
    """Test output dimensions of HOG1DTransformer.

    The time series length should always be num_bins*num_intervals
    (num_intervals is 2 by default)
    """
    X = np.ones(shape=(10, 1, 13))
    h = HOG1DTransformer(n_bins=num_bins).fit(X)
    res = h.transform(X)

    # get the dimension of the generated numpy array.
    n_cases, n_channels, n_timepoints = res.shape

    assert n_timepoints == corr_n_timepoints
    assert n_cases == 10
    assert n_channels == 1


@pytest.mark.parametrize("bad_num_intervals", ["str", 1.2, -1.2, -1, {}, 11, 0])
def test_bad_num_intervals(bad_num_intervals):
    """Check that exception is raised for bad num intervals.

    input types - string, float, negative int, negative float, empty dict
    and an int that is larger than the time series length.
    correct input is meant to be a positive integer of 1 or more.
    """
    X = np.ones(shape=(10, 1, 10))

    if not isinstance(bad_num_intervals, int):
        with pytest.raises(TypeError):
            HOG1DTransformer(n_intervals=bad_num_intervals).fit(X).transform(X)
    else:
        with pytest.raises(ValueError):
            HOG1DTransformer(n_intervals=bad_num_intervals).fit(X).transform(X)


@pytest.mark.parametrize("bad_scaling_factor", ["str", 1.2, -1.2, -1, {}, 0])
def test_bad_scaling_factor(bad_scaling_factor):
    """Check that exception is raised for bad scaling factor.

    input types - string, float, negative float, negative int,
    empty dict and zero.
    correct input is meant to be any number (so the floats and
    ints shouldn't raise an error).
    """
    X = np.ones(shape=(10, 1, 10))

    if not isinstance(bad_scaling_factor, numbers.Number):
        with pytest.raises(TypeError):
            HOG1DTransformer(scaling_factor=bad_scaling_factor).fit(X).transform(X)
    else:
        HOG1DTransformer(scaling_factor=bad_scaling_factor).fit(X).transform(X)


def test_output_of_transformer():
    """Check the transformer has changed the data correctly."""
    X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]])
    h = HOG1DTransformer().fit(X)
    res = h.transform(X)
    orig = np.array(
        [
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        ]
    )
    np.testing.assert_array_equal(res, orig)
    X = np.array([[[-5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2]]])
    h = h.fit(X)
    res = h.transform(X)
    orig = np.array([[[0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0]]])
    np.testing.assert_array_equal(res, orig)
