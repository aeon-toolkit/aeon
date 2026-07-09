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
    """Test multivariate dimension selection respects max_dims.

    dim_threshold=0 keeps every channel eligible, so the random truncation
    down to max_dims dimensions must run.
    """
    X, y = make_example_3d_numpy(n_cases=20, n_channels=10, n_timepoints=50)
    tde = IndividualTDE(max_dims=1, dim_threshold=0.0, random_state=0)
    tde._fit(X, y)
    assert len(tde._dims) == 1


def test_tde_pickle():
    """Test that a fitted IndividualTDE round-trips through pickle.

    The array-backed bags should pickle natively and the unpickled model
    should produce the same predictions as the original.
    """
    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50)
    tde = IndividualTDE(random_state=0)
    tde.fit(X, y)

    unpickled_tde = pickle.loads(pickle.dumps(tde))

    assert isinstance(unpickled_tde, IndividualTDE)
    np.testing.assert_array_equal(tde.predict(X), unpickled_tde.predict(X))


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


def test_tde_loocv_train_estimate_and_predict():
    """Test the loocv train estimate path and ensemble prediction.

    fit_predict_proba with the default "loocv" method exercises the stored
    per-member train predictions; predict and predict_proba are then checked
    on the fitted ensemble.
    """
    X_train, y_train = load_unit_test(split="train")

    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        train_estimate_method="loocv",
        random_state=0,
    )
    train_proba = tde.fit_predict_proba(X_train, y_train)

    assert train_proba.shape == (len(X_train), 2)
    np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)

    proba = tde.predict_proba(X_train)
    preds = tde.predict(X_train)

    assert proba.shape == (len(X_train), 2)
    np.testing.assert_almost_equal(proba.sum(axis=1), 1, decimal=4)
    assert all(p in tde.classes_ for p in preds)


def test_tde_deprecated_parameters_warn():
    """Test the deprecated alphabet_size and typed_dict parameters.

    Both parameters have no effect and raise a FutureWarning when a value is
    passed; the defaults must stay silent. TODO remove in v1.7.0 along with
    the parameters.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        IndividualTDE()
        TemporalDictionaryEnsemble()

    with pytest.warns(FutureWarning, match="alphabet_size"):
        IndividualTDE(alphabet_size=4)
    with pytest.warns(FutureWarning, match="typed_dict"):
        IndividualTDE(typed_dict=True)
    with pytest.warns(FutureWarning, match="typed_dict"):
        TemporalDictionaryEnsemble(typed_dict=False)


def test_kernel_ridge_parameter_selection_matches_sklearn():
    """Test the numpy kernel ridge helper against the sklearn original.

    The ensemble's guided parameter selection replaced StandardScaler +
    KernelRidge(kernel="poly", degree=1) with a direct numpy computation;
    the predictions must match sklearn's to numerical precision.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler

    from aeon.classification.dictionary_based._tde import _kernel_ridge_preds

    rng = np.random.RandomState(0)
    x_hist = rng.randint(0, 20, size=(30, 5)).astype(np.float64)
    x_hist[:, 3] = 1.0  # constant column, exercises the zero-std guard
    y_hist = rng.rand(30)
    candidates = rng.randint(0, 20, size=(40, 5)).astype(np.float64)

    scaler = StandardScaler().fit(x_hist)
    gp = KernelRidge(kernel="poly", degree=1)
    gp.fit(scaler.transform(x_hist), y_hist)
    expected = gp.predict(scaler.transform(candidates))

    actual = _kernel_ridge_preds(x_hist, y_hist, candidates)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_individual_train_acc_fallback_matches_symmetric_kernel(monkeypatch):
    """Test the per-case LOOCV fallback agrees with the symmetric kernel.

    Above _SYMMETRIC_LOOCV_MAX_N cases the ensemble falls back to per-case
    nearest neighbour searches instead of materialising the n x n similarity
    matrix. Forcing the threshold to zero must not change the accuracy
    estimate or the resulting ensemble behaviour.
    """
    from aeon.classification.dictionary_based import _tde

    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50)

    def fit_ensemble():
        tde = TemporalDictionaryEnsemble(
            n_parameter_samples=4,
            max_ensemble_size=2,
            randomly_selected_params=2,
            random_state=0,
        )
        tde.fit(X, y)
        return tde

    fast = fit_ensemble()
    monkeypatch.setattr(_tde, "_SYMMETRIC_LOOCV_MAX_N", 0)
    slow = fit_ensemble()

    assert [e._accuracy for e in fast.estimators_] == [
        e._accuracy for e in slow.estimators_
    ]
    np.testing.assert_array_equal(fast.predict(X), slow.predict(X))


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
