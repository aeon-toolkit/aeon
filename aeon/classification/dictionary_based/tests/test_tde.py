"""TDE test code."""

import pickle

import numpy as np
import pytest
from numba import types
from numba.typed import Dict

from aeon.classification.dictionary_based._tde import (
    IndividualTDE,
    TemporalDictionaryEnsemble,
    histogram_intersection,
)
from aeon.datasets import load_unit_test
from aeon.testing.data_generation import make_example_3d_numpy


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
    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50)
    with pytest.raises(ValueError, match="Invalid train_estimate_method"):
        tde.fit_predict_proba(X, y)
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        min_window=100,
        random_state=0,
        bigrams=True,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tde.fit(X, y)
        assert tde._min_window == 50


def test_tde_multivariate():
    """Test TDE with incorrect input."""
    # train TDE
    X, y = make_example_3d_numpy(n_cases=20, n_channels=10, n_timepoints=50)
    tde = IndividualTDE(max_dims=1)
    tde._fit(X, y)
    assert len(tde._dims) == 1


def test_tde_pickle():
    """Test the dict conversion work around for pickle."""
    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50)
    tde = IndividualTDE(typed_dict=True)
    tde.fit(X, y)
    pickled_tde = pickle.dumps(tde)
    unpickled_tde = pickle.loads(pickled_tde)
    assert isinstance(unpickled_tde, IndividualTDE)


def test_histogram_intersection():
    """Test the histogram intersection function used by TDE."""
    first = np.array([1, 0, 0, 1, 0])
    second = np.array([1, 2, 3, 5, 10])
    res = histogram_intersection(first, second)
    assert res == 2
    first = {1: 1, 4: 1}
    second = {1: 1, 2: 2, 3: 3, 4: 5, 5: 10}
    res = histogram_intersection(first, second)
    assert res == 2
    numba_first = Dict.empty(key_type=types.int64, value_type=types.int64)
    for key, value in first.items():
        numba_first[key] = value
    numba_second = Dict.empty(key_type=types.int64, value_type=types.int64)
    for key, value in second.items():
        numba_second[key] = value

    res = histogram_intersection(numba_first, numba_second)
    assert res == 2


def test_subsampling_in_highly_imbalanced_datasets():
    """Test the subsampling during fit for highly imbalanced datasets.

    This test case tests the fix for bug #1726.
    https://github.com/aeon-toolkit/aeon/issues/1726
    """
    X = np.random.rand(10, 1, 20)
    y_sc = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    tde = TemporalDictionaryEnsemble(random_state=42)
    tde.fit(X, y_sc)

    assert tde.is_fitted
