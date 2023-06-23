# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Test input for the BaseCollectionTransformer."""

__author__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest

from aeon.datatypes import convert_to
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
    Xt = t.fit_transform(X, y)

    assert isinstance(Xt, list)
    assert isinstance(Xt[0], np.ndarray)
    assert Xt[0].ndim == 2


@pytest.mark.parametrize("dtype", ["pd.Series", "pd.DataFrame"])
def test_collection_transformer_invalid_input(dtype):
    """Test that BaseCollectionTransformer fails with series input."""
    y = convert_to(
        _make_series(),
        to_type=dtype,
    )

    t = _Dummy()

    with pytest.raises(TypeError):
        t.fit_transform(y)


@pytest.mark.parametrize("dtype", ["pd.Series", "pd.DataFrame", "np.ndarray"])
def test_collection_transformer_wrapper_series(dtype):
    """Test that the wrapper for regular transformers works with series input."""
    y = convert_to(
        _make_series(),
        to_type=dtype,
    )

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


class _Dummy(BaseCollectionTransformer):
    """Dummy transformer for testing.

    Converts a numpy array to a list of numpy arrays.
    """

    _tags = {"X_inner_mtype": ["numpy3D", "np-list"]}

    def __init__(self):
        super(_Dummy, self).__init__()

    def _transform(self, X, y=None):
        assert isinstance(X, np.ndarray) or isinstance(X, list)
        if isinstance(X, np.ndarray):
            return [x for x in X]
        return X
