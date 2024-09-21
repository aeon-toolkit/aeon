"""Test the base series transformer."""

import numpy as np
import pytest

from aeon.testing.mock_estimators import (
    MockMultivariateSeriesTransformer,
    MockSeriesTransformerNoFit,
    MockUnivariateSeriesTransformer,
)

ALL_TRANSFORMERS = [MockMultivariateSeriesTransformer(), MockSeriesTransformerNoFit()]
MULTIVARIATE_SERIES = [
    np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
    np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
    np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
]


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS)
@pytest.mark.parametrize("series", MULTIVARIATE_SERIES)
def test_fit_transform_multivariate(transformer, series):
    """Test fit then transform equivalent to fit_transform with setting axis."""
    transformer.fit(series)
    x2 = transformer.transform(series)
    x3 = transformer.fit_transform(series)
    np.testing.assert_array_almost_equal(x2, x3)
    assert series.shape == x2.shape


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS)
@pytest.mark.parametrize("series", MULTIVARIATE_SERIES)
def test_axis(transformer, series):
    """Test axis."""
    transformer.fit(series, axis=1)
    x2 = transformer.transform(series, axis=1)
    assert series.shape == x2.shape


def test_fit_transform_univariate():
    """Test fit transform for univariate."""
    x1 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    transformer = MockUnivariateSeriesTransformer()
    x2 = transformer.fit_transform(x1)
    assert x1.shape == x2.shape
    x2 = transformer.fit_transform(x1, x2)
    x3 = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    x4 = transformer.fit_transform(x3)
    assert isinstance(x4, np.ndarray) and x4.ndim == 1
    with pytest.raises(ValueError, match="Multivariate data not supported"):
        transformer.fit_transform(np.random.rand(2, 10))
    transformer.fit(x1, x2)
    x3 = transformer.transform(x1, x2)
    x4 = transformer.fit_transform(x1, x2)
    np.testing.assert_array_almost_equal(x3, x4)
