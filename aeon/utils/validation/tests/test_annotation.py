#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Test functions in the _dependencies file."""
import pytest

from aeon.utils.validation.annotation import check_fmt, check_labels


def test_annotation():
    """Check fmt."""
    with pytest.raises(ValueError):
        check_fmt("Stupid function")
    with pytest.raises(ValueError):
        check_labels("Stupid function")
