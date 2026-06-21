"""Test the base series transformer."""

import numpy as np
import pytest

from aeon.testing.mock_estimators import (
    MockMultivariateSeriesTransformer,
    MockSeriesTransformerNoFit,
    MockUnivariateSeriesTransformer,
)
from aeon.transformations.series.base import BaseSeriesTransformer

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


class _RequiresYTransformer(BaseSeriesTransformer):
    """Minimal transformer that requires y and uses the default _fit."""

    _tags = {"requires_y": True, "fit_is_empty": False}

    def __init__(self):
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        return X


def test_fit_requires_y_raises_without_y():
    """Test fit raises a ValueError when requires_y is True and y is None."""
    transformer = _RequiresYTransformer()
    with pytest.raises(ValueError, match="requires_y is true"):
        transformer.fit(np.array([1.0, 2.0, 3.0]))


def test_fit_transform_requires_y_raises_without_y():
    """Test fit_transform raises a ValueError when requires_y is True, y is None."""
    transformer = _RequiresYTransformer()
    with pytest.raises(ValueError, match="requires_y is true"):
        transformer.fit_transform(np.array([1.0, 2.0, 3.0]))


def test_default_fit_is_a_noop():
    """Test the base _fit default implementation does nothing and succeeds."""
    transformer = _RequiresYTransformer()
    transformer.fit(np.array([1.0, 2.0, 3.0]), y=np.array([0]))
    assert transformer.is_fitted


def test_fit_transform_requires_y_succeeds_with_y():
    """Test fit_transform proceeds normally when requires_y is True and y given."""
    transformer = _RequiresYTransformer()
    Xt = transformer.fit_transform(np.array([1.0, 2.0, 3.0]), y=np.array([0]))
    np.testing.assert_array_equal(Xt, np.array([1.0, 2.0, 3.0]))


def test_postprocess_series_transposes_for_axis_zero():
    """Test the output is transposed when axis differs from the transformer's axis."""
    transformer = MockMultivariateSeriesTransformer()
    Xt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = transformer._postprocess_series(Xt, axis=0)
    np.testing.assert_array_equal(result, Xt.T)


def test_postprocess_series_with_axis_none_uses_self_axis():
    """Test _postprocess_series with axis=None falls back to the transformer's axis."""
    transformer = MockMultivariateSeriesTransformer()
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    Xt = transformer._postprocess_series(X, axis=None)
    np.testing.assert_array_equal(Xt, X)
