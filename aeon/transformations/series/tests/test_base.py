""""Test the base series transformer."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from aeon.testing.utils.data_gen import (
    make_1d_numpy_series,
    make_2d_numpy_series,
    make_example_3d_numpy,
    make_series,
)
from aeon.transformations.series import DummySeriesTransformer


@pytest.mark.parametrize(
    "data_gen",
    [
        make_1d_numpy_series,
        make_2d_numpy_series,
        make_series,
    ],
)
def test_series_transformer_valid_input(data_gen):
    """Test that BaseCollectionTransformer works with collection input."""
    X = data_gen()

    t = DummySeriesTransformer()
    t.fit(X)
    Xt = t.transform(X)
    assert isinstance(Xt, np.ndarray)
    assert Xt.ndim == 2


@pytest.mark.parametrize("data_gen", [make_example_3d_numpy])
def test_series_transformer_invalid_input(data_gen):
    """Test that BaseCollectionTransformer fails with series input."""
    X, y = data_gen()
    t = DummySeriesTransformer()
    with pytest.raises(ValueError):
        t.fit_transform(X, y)


def test_inverse_transform():
    """Test inverse transform."""
    d = DummySeriesTransformer()
    x = make_2d_numpy_series(axis=1)
    d.fit(x)
    d.set_tags(**{"skip-inverse-transform": True})
    x2 = d.inverse_transform(x)
    assert_almost_equal(x, x2)
    d.set_tags(**{"skip-inverse-transform": False})
    xt = d.transform(x)
    x3 = d.inverse_transform(xt)
    assert_almost_equal(x, x3)


def test_axis_convertion():
    """Test axis convertion."""
    # Expect shape=(n_channels, n_timepoints) (axis=1)
    d = DummySeriesTransformer()
    x = make_2d_numpy_series(axis=0)
    d.fit(x, axis=0)
    xt = d.transform(x, axis=0)
    expected = x.copy()
    for i in range(x.shape[1]):
        expected[:, i] = expected[:, i] + (d.constant + d.random_values_[i])

    assert_array_almost_equal(xt, expected)
    x2 = d.inverse_transform(xt, axis=0)
    assert_array_almost_equal(x, x2)
