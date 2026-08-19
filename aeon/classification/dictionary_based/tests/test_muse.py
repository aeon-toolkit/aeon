"""Test MUSE multivariate classifier."""

import pytest

from aeon.classification.dictionary_based import MUSE
from aeon.testing.data_generation import make_example_3d_numpy


def test_muse():
    """Test MUSE with first order differences and incorrect input."""
    muse = MUSE(use_first_order_differences=True)
    X, y = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=5)
    X2 = muse._add_first_order_differences(X)
    assert X2.shape[2] == X.shape[2] and X2.shape[1] == X.shape[1] * 2
    with pytest.raises(ValueError, match="Error in MUSE, min_window"):
        muse.fit(X, y)
