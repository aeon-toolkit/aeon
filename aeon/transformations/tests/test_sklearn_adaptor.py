"""Tests for TabularToSeriesAdaptor."""

__maintainer__ = []

import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

from aeon.datasets import load_airline
from aeon.transformations.adapt import TabularToSeriesAdaptor


def test_boxcox_transform():
    """Test whether adaptor based transformer behaves like the raw wrapped method."""
    y = load_airline()
    t = TabularToSeriesAdaptor(PowerTransformer(method="box-cox", standardize=False))
    actual = t.fit_transform(y)

    expected, _ = boxcox(np.asarray(y))  # returns fitted lambda as second output
    np.testing.assert_array_equal(actual, expected)
