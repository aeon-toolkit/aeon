"""Tests for SlidingWindowSegmenter."""

__authors__ = ["mloning"]


import numpy as np
import pytest

from aeon.transformations.collection.segment import SlidingWindowSegmenter


# Check that exception is raised for bad window length.
# input types - string, float, negative int, negative float and empty dict
# correct input is meant to be a positive integer of 1 or more.
@pytest.mark.parametrize("bad_window_length", ["str", 1.2, -1.2, -1, {}])
def test_bad_input_args(bad_window_length):
    """Test that bad inputs raise value error."""
    X = np.ones(shape=(10, 1, 10))
    if not isinstance(bad_window_length, int):
        with pytest.raises(TypeError):
            SlidingWindowSegmenter(window_length=bad_window_length).fit_transform(X)
    else:
        with pytest.raises(ValueError):
            SlidingWindowSegmenter(window_length=bad_window_length).fit_transform(X)


# Check the transformer has changed the data correctly.
def test_output_of_transformer():
    """Test correct output of SlidingWindowSegmenter."""
    X = np.array([[[1, 2, 3, 4, 5, 6]]])
    st = SlidingWindowSegmenter(window_length=1).fit(X)
    res = st.transform(X)
    orig = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
    np.testing.assert_array_equal(res, orig)

    st = SlidingWindowSegmenter(window_length=5).fit(X)
    res = st.transform(X)
    orig = np.array(
        [
            [
                [1.0, 1.0, 1.0, 2.0, 3.0],
                [1.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 6.0],
                [4.0, 5.0, 6.0, 6.0, 6.0],
            ]
        ]
    )
    np.testing.assert_array_equal(res, orig)

    st = SlidingWindowSegmenter(window_length=10).fit(X)
    res = st.transform(X)
    orig = np.array(
        [
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            ]
        ]
    )
    np.testing.assert_array_equal(res, orig)


@pytest.mark.parametrize(
    "time_n_timepoints,window_length", [(5, 1), (10, 5), (15, 9), (20, 13), (25, 19)]
)
def test_output_dimensions(time_n_timepoints, window_length):
    """Test output dimensions of SlidingWindowSegmenter."""
    X = np.ones(shape=(10, 1, time_n_timepoints))
    st = SlidingWindowSegmenter(window_length=window_length)
    res = st.fit_transform(X)

    # get the dimension of the generated dataframe.
    n_cases, n_channels, n_timepoints = res.shape

    assert n_timepoints == window_length
    assert n_cases == 10
    assert n_channels == time_n_timepoints
