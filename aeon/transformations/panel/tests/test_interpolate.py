# -*- coding: utf-8 -*-
"""Tests for TSInterpolator."""
import numpy as np

from aeon.transformations.panel.interpolate import TSInterpolator


def test_interpolator():
    """Test TSInterpolator resizing."""

    X_list = []
    for i in range(10):
        X_list.append(np.random.rand(5, 10 + i))
    trs = TSInterpolator(length=50)
    X_new = trs.fit_transform(X_list)
    assert X_new.shape == (10, 5, 50)
    X_array = np.random.rand(10, 3, 30)
    X_new = trs.fit_transform(X_array)
    assert X_new.shape == (10, 3, 50)
