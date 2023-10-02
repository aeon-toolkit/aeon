#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Test functions in the init file."""

import pytest

from aeon.utils.validation import check_n_jobs, check_window_length


@pytest.mark.parametrize(
    "window_length, n_timepoints, expected",
    [
        (0.2, 33, 7),
        (43, 23, 43),
        (33, 1, 33),
        (33, None, 33),
        (None, 19, None),
        (None, None, None),
        (67, 0.3, 67),  # bad arg
    ],
)
def test_check_window_length(window_length, n_timepoints, expected):
    assert check_window_length(window_length, n_timepoints) == expected


@pytest.mark.parametrize(
    "window_length, n_timepoints",
    [
        ("string", 34),
        ("string", "string"),
        (6.2, 33),
        (-5, 34),
        (-0.5, 15),
        (6.1, 0.3),
        (0.3, 0.1),
        (-2.4, 10),
        (0.2, None),
    ],
)
def test_window_length_bad_arg(window_length, n_timepoints):
    with pytest.raises(ValueError):
        check_window_length(window_length, n_timepoints)


def test_check_n_jobs():
    """Test check_n_jobs."""
    assert check_n_jobs(None) == check_n_jobs(0) == 1
    with pytest.raises(ValueError, match="must be None or an integer"):
        check_n_jobs("Arsenal")
    res = check_n_jobs(-1)
    assert res >= 1
