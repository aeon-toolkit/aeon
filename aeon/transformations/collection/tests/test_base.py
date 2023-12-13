"""Test input for the BaseCollectionTransformer."""

__author__ = ["MatthewMiddlehurst", "TonyBagnall"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.transformations.collection import (
    BaseCollectionTransformer,
    CollectionToSeriesWrapper,
)
from aeon.utils._testing.collection import (
    make_2d_test_data,
    make_3d_test_data,
    make_unequal_length_test_data,
)
from aeon.utils._testing.series import _make_series


@pytest.mark.parametrize(
    "data_gen", [make_3d_test_data, make_2d_test_data, make_unequal_length_test_data]
)
def test_collection_transformer_valid_input(data_gen):
    """Test that BaseCollectionTransformer works with collection input."""
    X, y = data_gen()

    t = _Dummy()
    t.fit(X, y)
    Xt = t.transform(X)

    assert isinstance(Xt, list)
    assert isinstance(Xt[0], np.ndarray)
    assert Xt[0].ndim == 2


@pytest.mark.parametrize("dtype", ["pd.Series"])
def test_collection_transformer_invalid_input(dtype):
    """Test that BaseCollectionTransformer fails with series input."""
    y = (_make_series(),)
    t = _Dummy()
    with pytest.raises(TypeError):
        t.fit_transform(y)


@pytest.mark.parametrize("dtype", ["pd.Series", "pd.DataFrame", "np.ndarray"])
def test_collection_transformer_wrapper_series(dtype):
    """Test that the wrapper for regular transformers works with series input."""
    if dtype == "np.ndarray":
        y = _make_series(return_numpy=True)
    else:
        y = _make_series()
        if dtype == "pd.DataFrame":
            y = y.to_frame(name="series1")
    wrap = CollectionToSeriesWrapper(transformer=_Dummy())
    yt = wrap.fit_transform(y)

    assert isinstance(yt, list)
    assert isinstance(yt[0], np.ndarray)
    assert len(yt) == 1
    assert yt[0].ndim == 2


@pytest.mark.parametrize("data_gen", [make_3d_test_data, make_unequal_length_test_data])
def test_collection_transformer_wrapper_collection(data_gen):
    """Test that the wrapper for regular transformers works with collection input."""
    X, y = data_gen()

    wrap = CollectionToSeriesWrapper(transformer=_Dummy())
    Xt = wrap.fit_transform(X, y)

    assert isinstance(Xt, list)
    assert isinstance(Xt[0], np.ndarray)
    assert Xt[0].ndim == 2


def test_inverse_transform():
    """Test inverse transform."""
    d = _Dummy()
    x, _ = make_3d_test_data()
    d.fit(x)
    d.set_tags(**{"skip-inverse-transform": True})
    x2 = d.inverse_transform(x)
    assert_almost_equal(x, x2)
    d.set_tags(**{"skip-inverse-transform": False})
    with pytest.raises(
        NotImplementedError, match="does not implement " "inverse_transform"
    ):
        d.inverse_transform(x)
    d.set_tags(**{"capability:inverse_transform": True})
    x2 = d.inverse_transform(x)
    assert_almost_equal(x, x2)


class _Dummy(BaseCollectionTransformer):
    """Dummy transformer for testing.

    Converts a numpy array to a list of numpy arrays.
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self):
        super(_Dummy, self).__init__()

    def _transform(self, X, y=None):
        assert isinstance(X, np.ndarray) or isinstance(X, list)
        if isinstance(X, np.ndarray):
            return [x for x in X]
        return X

    def _inverse_transform(self, X, y=None):
        return X
