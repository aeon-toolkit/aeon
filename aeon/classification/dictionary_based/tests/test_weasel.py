"""WEASEL test code."""

from aeon.classification.dictionary_based._weasel_v2 import WEASEL_V2
from aeon.datasets import *

# def test_weasel_score():
#     """Test of WEASEL train estimate on unit test data."""
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train")
#     X_test, y_test = load_unit_test(split="test")
#
#     # train weasel
#     weasel = WEASEL(random_state=0)
#     weasel.fit(X_train, y_train)
#     score = weasel.score(X_test, y_test)
#
#     assert isinstance(score, float)
#     np.testing.assert_almost_equal(score, 0.727272, decimal=4)


def test_weasel_v2_score():
    """Test of WEASEL v2 train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_acsf1(split="train")
    X_test, y_test = load_acsf1(split="test")

    # train weasel
    weasel = WEASEL_V2(random_state=0)
    weasel.fit(X_train, y_train)
    score = weasel.score(X_test, y_test)

    print(f"Weasel v2 Score {score*100:0.1f}")
    assert isinstance(score, float)
    # np.testing.assert_almost_equal(score, 0.8685, decimal=4)


def test_weasel_v3_score():
    """Test of WEASEL v3 train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_acsf1(split="train")
    X_test, y_test = load_acsf1(split="test")

    # train weasel
    weasel = WEASEL_V2(
        random_state=0,
        alphabet_allocation_method="linear_scale",
        feature_selection_strategy="pca",
    )
    weasel.fit(X_train, y_train)
    score = weasel.score(X_test, y_test)

    assert isinstance(score, float)

    print(f"Weasel v3 Score {score*100:0.1f}")
    # np.testing.assert_almost_equal(score, 0.834, decimal=4)
