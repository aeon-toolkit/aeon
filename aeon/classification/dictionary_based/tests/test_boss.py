"""BOSS test code."""

from aeon.classification.dictionary_based import ContractableBOSS
from aeon.testing.utils.data_gen import make_2d_test_data


def test_cboss_small_train():
    """Test with a small amount of train cases, subsampling can cause issues."""
    X, y = make_2d_test_data(n_cases=3, n_timepoints=20, n_labels=2)
    cboss = ContractableBOSS(n_parameter_samples=10, max_ensemble_size=3)
    cboss.fit(X, y)
    cboss.predict(X)
