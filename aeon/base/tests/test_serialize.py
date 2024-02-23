"""Tests for serialisation."""

import os

import pytest

from aeon.base import load
from aeon.classification import DummyClassifier
from aeon.testing.utils.data_gen import make_example_3d_numpy


def test_save_and_load():
    """Test save and load."""
    X, y = make_example_3d_numpy()
    dummy = DummyClassifier()
    dummy.fit(X, y)
    loc = "testy"
    dummy.save(loc)
    assert os.path.isfile(loc + ".zip")
    with pytest.raises(
        TypeError, match="expected to either be a string or a Path " "object"
    ):
        dummy.save(dummy)
    loaded = load(loc)
    assert loaded.is_fitted
    os.remove(loc + ".zip")
