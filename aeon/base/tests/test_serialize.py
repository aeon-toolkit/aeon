# -*- coding: utf-8 -*-
"""Test save and load."""
import os

import pytest

from aeon.base import load
from aeon.classification import DummyClassifier
from aeon.utils._testing.collection import make_3d_test_data


def test_save_and_load():
    X, y = make_3d_test_data()
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
