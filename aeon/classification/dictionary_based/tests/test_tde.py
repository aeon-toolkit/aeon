"""TDE test code."""

import numpy as np

from aeon.classification.dictionary_based._tde import TemporalDictionaryEnsemble
from aeon.datasets import load_unit_test


def test_tde_oob_train_estimate():
    """Test of TDE oob train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        train_estimate_method="oob",
        random_state=0,
    )
    train_proba = tde.fit_predict_proba(X_train, y_train)

    assert isinstance(train_proba, np.ndarray)
    assert train_proba.shape == (len(X_train), 2)
    np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)
