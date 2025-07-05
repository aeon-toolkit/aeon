"""Test signatures helper functions."""

import numpy as np

from aeon.transformations.collection.signature_based._rescaling import (
    _rescale_path,
    _rescale_signature,
)


def test_rescale_path():
    """Test rescale_path function."""
    X = np.random.random((10, 2, 40))
    data = np.swapaxes(X, 1, 2)
    trans = _rescale_path(data, 1)
    assert isinstance(trans, np.ndarray)


def test_rescale_signature():
    """Test rescale_signatures."""
    X = np.random.random((10, 2, 40))
    data = np.swapaxes(X, 1, 2)
    trans = _rescale_signature(data, 1)
    assert isinstance(trans, np.ndarray)
