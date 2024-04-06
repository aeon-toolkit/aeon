"""Test the base series transformer."""

import numpy as np
import pytest

from aeon.testing.mock_estimators import (
    MockSeriesTransformer,
    MockSeriesTransformerNoFit,
)

ALL_TRANSFORMERS = [MockSeriesTransformer(), MockSeriesTransformerNoFit()]
SERIES = [
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
    np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
    np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
]


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS)
@pytest.mark.parametrize("series", SERIES)
def test_fit_transform(transformer, series):
    """Test fit then transform equivalent to fit_transform with setting axis."""
    transformer.fit(series)
    x2 = transformer.transform(series)
    x3 = transformer.fit_transform(series)
    np.testing.assert_array_almost_equal(x2, x3)


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS)
@pytest.mark.parametrize("series", SERIES)
def test_axis(transformer, series):
    """Test axis."""
    transformer.fit(series, axis=1)
    x2 = transformer.transform(series, axis=1)
    assert series.shape == x2.shape
