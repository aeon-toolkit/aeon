"""TDE test code."""

import numpy as np
import pytest
from aeon.classification.dictionary_based._tde import TemporalDictionaryEnsemble, IndividualTDE
from aeon.datasets import load_unit_test
from aeon.testing.data_generation import make_example_3d_numpy
import pickle

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


def test_tde_incorrect_input():
    """Test TDE with incorrect input."""
    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        train_estimate_method="FOOBAR",
        random_state=0,
    )
    X, y =make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50)
    with pytest.raises(ValueError, match="Invalid train_estimate_method"):
        tde.fit_predict_proba(X, y)
    tde = IndividualTDE(typed_dict=True)
    tde.fit(X,y)
    pickled_data = pickle.dumps(tde)
    unpickled_data = pickle.loads(pickled_data)
