"""Test MUSE multivariate classifier."""

import numpy as np
import pytest

from aeon.classification.dictionary_based import MUSE
from aeon.datasets import load_basic_motions
from aeon.testing.data_generation import make_example_3d_numpy


def test_muse():
    """Test MUSE with first order differences and incorrect input."""
    muse = MUSE(use_first_order_differences=True)
    X, y = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=5)
    X2 = muse._add_first_order_differences(X)
    assert X2.shape[2] == X.shape[2] and X2.shape[1] == X.shape[1] * 2
    with pytest.raises(ValueError, match="Error in MUSE, min_window"):
        muse.fit(X, y)


def test_muse_score():
    """Test of MUSE train estimate on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    # train muse
    muse = MUSE(random_state=0)
    muse.fit(X_train, y_train)
    score = muse.score(X_test, y_test)

    assert isinstance(score, float)
    np.testing.assert_almost_equal(score, 1.0, decimal=4)
