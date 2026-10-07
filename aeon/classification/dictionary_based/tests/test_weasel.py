"""WEASEL test code."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV

from aeon.classification.dictionary_based._weasel import WEASEL
from aeon.classification.dictionary_based._weasel_v2 import (
    WEASEL_V2,
    WEASELTransformerV2,
)
from aeon.datasets import load_unit_test


def test_weasel_score():
    """Test of WEASEL train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    # train weasel
    weasel = WEASEL(random_state=0)
    weasel.fit(X_train, y_train)
    score = weasel.score(X_test, y_test)

    assert isinstance(score, float)
    np.testing.assert_almost_equal(score, 0.727272, decimal=4)


def test_weasel_v2_score():
    """Test of WEASEL v2 train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    # train weasel
    weasel = WEASEL_V2(random_state=0)
    weasel.fit(X_train, y_train)
    score = weasel.score(X_test, y_test)

    assert isinstance(score, float)
    np.testing.assert_almost_equal(score, 0.90909, decimal=4)


def test_weasel_v2_transform_no_y_unsupervised():
    """Test of WEASELTransformerV2 without y."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train weasel
    weasel = WEASELTransformerV2(random_state=0, feature_selection="none")
    all_words = weasel.fit_transform(X_train)

    np.testing.assert_equal(len(all_words), X_train.shape[0])


def test_weasel_v2_transform_float_output():
    """Test WEASELTransformerV2 output is a float dtype.

    Some versions of sklearn RidgeClassifierCV incorrectly convert values
    using the input dtype, which can cause issues.
    """
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    weasel = WEASELTransformerV2(random_state=0)
    Xt_train = weasel.fit_transform(X_train, y_train)
    Xt_test = weasel.transform(X_test)

    assert Xt_train.dtype == np.float32
    assert Xt_test.dtype == np.float32

    clf = RidgeClassifierCV(alphas=np.logspace(-1, 5, 10)).fit(Xt_train, y_train)
    assert not np.allclose(clf.coef_, 0)
    np.testing.assert_almost_equal(clf.score(Xt_test, y_test), 0.90909, decimal=4)


def test_weasel_v2_transform_no_y_supervised():
    """Test of WEASELTransformerV2 without y."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train weasel
    weasel = WEASELTransformerV2(random_state=0, feature_selection="chi2_top_k")

    with pytest.raises(
        ValueError,
        match="Class values must be provided for chi2_top_k feature selection.",
    ):
        weasel.fit_transform(X_train)  # Should fail here
