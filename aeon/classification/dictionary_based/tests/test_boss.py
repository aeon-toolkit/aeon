"""BOSS test code."""

import pytest

from aeon.classification.dictionary_based import BOSSEnsemble
from aeon.testing.data_generation import (
    make_example_3d_numpy,
)


def test_boss_min_window():
    """Test BOSS throws error when min window too big."""
    boss = BOSSEnsemble(min_window=20)
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=10)
    with pytest.raises(ValueError, match="Error in BOSSEnsemble, min_window"):
        boss._fit(X, y)
