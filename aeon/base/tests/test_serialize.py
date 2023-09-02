# -*- coding: utf-8 -*-
"""Test function for load in _serialize."""
import os

from aeon.base import load
from aeon.classification import DummyClassifier
from aeon.utils._testing.collection import make_3d_test_data


def test_load():
    X, y = make_3d_test_data()
    dummy = DummyClassifier()
    dummy.fit(X, y)
    loc = "testy"
    dummy.save(loc)
    loaded = load(loc)
    assert loaded.is_fitted
    if os.path.isfile(loc + ".zip"):
        os.remove(loc + ".zip")
