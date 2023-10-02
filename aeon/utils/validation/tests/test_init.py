#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Test functions in the init file."""

import pytest

from aeon.utils.validation import check_n_jobs


def test_check_n_jobs():
    """Test check_n_jobs."""
    assert check_n_jobs(None) == check_n_jobs(0) == 1
    with pytest.raises(ValueError, match="must be None or an integer"):
        check_n_jobs("Arsenal")
    res = check_n_jobs(-1)
    assert res >= 1
