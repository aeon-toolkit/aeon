"""Tests for the CollectionToSeriesWrapper transformer."""

import numpy as np

from aeon.transformations.collection.base import (
    BaseCollectionTransformer,
    CollectionInverseTransformerMixin,
)
from aeon.transformations.series import CollectionToSeriesWrapper


class _StatefulCollectionTransformer(
    CollectionInverseTransformerMixin, BaseCollectionTransformer
):
    """Collection transformer with non-empty fit, scales by a fitted factor."""

    def __init__(self, factor=2.0):
        self.factor = factor
        super().__init__()

    def _fit(self, X, y=None):
        self.fitted_factor_ = self.factor
        return self

    def _transform(self, X, y=None):
        return X * self.fitted_factor_

    def _inverse_transform(self, X, y=None):
        return X / self.fitted_factor_


class _StatelessCollectionTransformer(
    CollectionInverseTransformerMixin, BaseCollectionTransformer
):
    """Collection transformer with an empty fit (fit_is_empty=True)."""

    _tags = {
        "fit_is_empty": True,
    }

    def __init__(self, factor=3.0):
        self.factor = factor
        super().__init__()

    def _transform(self, X, y=None):
        return X * self.factor

    def _inverse_transform(self, X, y=None):
        return X / self.factor


def test_collection_wrapper_fit_and_transform_stateful():
    """Test fit/transform uses the fitted collection_transformer_ instance."""
    X = np.array([[1.0, 2.0, 3.0]])

    wrapper = CollectionToSeriesWrapper(_StatefulCollectionTransformer(factor=2.0))
    wrapper.fit(X)
    Xt = wrapper.transform(X)

    np.testing.assert_array_equal(Xt, (X * 2.0).squeeze())


def test_collection_wrapper_transform_with_fit_is_empty():
    """Test transform uses self.transformer directly when fit_is_empty is True."""
    X = np.array([[1.0, 2.0, 3.0]])

    wrapper = CollectionToSeriesWrapper(_StatelessCollectionTransformer(factor=3.0))
    wrapper.fit(X)
    Xt = wrapper.transform(X)

    np.testing.assert_array_equal(Xt, (X * 3.0).squeeze())


def test_collection_wrapper_fit_transform():
    """Test fit_transform clones and fits the transformer, then transforms."""
    X = np.array([[1.0, 2.0, 3.0]])

    wrapper = CollectionToSeriesWrapper(_StatefulCollectionTransformer(factor=2.0))
    Xt = wrapper.fit_transform(X)

    np.testing.assert_array_equal(Xt, (X * 2.0).squeeze())


def test_collection_wrapper_inverse_transform_stateful():
    """Test inverse_transform uses the fitted collection_transformer_ instance."""
    X = np.array([[1.0, 2.0, 3.0]])

    wrapper = CollectionToSeriesWrapper(_StatefulCollectionTransformer(factor=2.0))
    wrapper.fit(X)
    Xt = wrapper.transform(X)
    Xinv = wrapper.inverse_transform(Xt)

    np.testing.assert_allclose(Xinv, X.squeeze())


def test_collection_wrapper_inverse_transform_with_fit_is_empty():
    """Test inverse_transform uses self.transformer when fit_is_empty is True.

    The wrapper's own fit short-circuits without fitting anything when
    fit_is_empty is True, so the inner transformer needs to already be fitted
    by the caller for its (always fit-checked) inverse_transform to succeed.
    """
    X = np.array([[1.0, 2.0, 3.0]])

    inner = _StatelessCollectionTransformer(factor=3.0)
    inner.fit(np.zeros((1, 1, 3)))

    wrapper = CollectionToSeriesWrapper(inner)
    wrapper.fit(X)
    Xt = wrapper.transform(X)
    Xinv = wrapper.inverse_transform(Xt)

    np.testing.assert_allclose(Xinv, X.squeeze())


def test_collection_wrapper_get_test_params():
    """Test default test parameters are valid and usable."""
    params = CollectionToSeriesWrapper._get_test_params()
    wrapper = CollectionToSeriesWrapper(**params)
    assert isinstance(wrapper, CollectionToSeriesWrapper)
