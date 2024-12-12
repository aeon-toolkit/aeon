"""Test input for the BaseCollectionTransformer."""

__maintainer__ = []

import numpy as np
import pytest

from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_pandas_series,
)
from aeon.transformations.collection import BaseCollectionTransformer


@pytest.mark.parametrize(
    "data_gen",
    [
        make_example_3d_numpy,
        make_example_2d_numpy_collection,
        make_example_3d_numpy_list,
    ],
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
    y = (make_example_pandas_series(),)
    t = _Dummy()
    with pytest.raises(TypeError):
        t.fit_transform(y)


def test_raise_inverse_transform():
    """Test that inverse transform raises NotImplementedError."""
    d = _Dummy()
    x, _ = make_example_3d_numpy()
    d.fit(x)
    with pytest.raises(
        NotImplementedError, match="does not implement " "inverse_transform"
    ):
        d.inverse_transform(x)


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
        super().__init__()

    def _transform(self, X, y=None):
        assert isinstance(X, np.ndarray) or isinstance(X, list)
        if isinstance(X, np.ndarray):
            return [x for x in X]
        return X

    def _inverse_transform(self, X, y=None):
        return X
