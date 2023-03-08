# -*- coding: utf-8 -*-
"""WEASEL test code."""
import numpy as np

from sktime.classification.dictionary_based._weasel import WEASEL
from sktime.classification.dictionary_based._weasel_v2 import WEASEL_V2
from sktime.datasets import load_unit_test


def test_weasel_train_estimate():
    """Test of WEASEL train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    # train weasel
    weasel = WEASEL(random_state=0)
    weasel.fit(X_train, y_train)
    score = weasel.score(X_test, y_test)

    assert isinstance(score, np.float_)
    np.testing.assert_almost_equal(score, 0.727272, decimal=4)


def test_weasel_v2_train_estimate():
    """Test of WEASEL v2 train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    # train weasel
    weasel = WEASEL_V2(random_state=0)
    weasel.fit(X_train, y_train)
    score = weasel.score(X_test, y_test)

    assert isinstance(score, np.float_)
    np.testing.assert_almost_equal(score, 0.90909, decimal=4)
