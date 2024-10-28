"""BOSS test code."""

import pytest

from aeon.classification.dictionary_based import BOSSEnsemble, ContractableBOSS
from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)


def test_cboss_small_train():
    """Test with a small amount of train cases, subsampling can cause issues."""
    X, y = make_example_2d_numpy_collection(n_cases=3, n_timepoints=20, n_labels=2)
    cboss = ContractableBOSS(n_parameter_samples=10, max_ensemble_size=3)
    cboss.fit(X, y)
    cboss.predict(X)


def test_boss_min_window():
    """Test BOSS throws error when min window too big."""
    boss = BOSSEnsemble(min_window=20)
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=10)
    with pytest.raises(ValueError, match="Error in BOSSEnsemble, min_window"):
        boss._fit(X, y)
