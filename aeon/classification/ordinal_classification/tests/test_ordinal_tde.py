"""Test ordinal TDE."""

import pytest

from aeon.classification.ordinal_classification import IndividualOrdinalTDE, OrdinalTDE
from aeon.testing.data_generation import make_example_3d_numpy


def test_ordinal_tde_incorrect_input():
    """Test Ordinal TDE with incorrect input."""
    # train TDE
    tde = OrdinalTDE(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        train_estimate_method="FOOBAR",
        random_state=0,
    )
    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50)
    with pytest.raises(ValueError, match="Invalid train_estimate_method"):
        tde.fit_predict_proba(X, y)
    tde = OrdinalTDE(
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
    tde = OrdinalTDE(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        train_estimate_method="oob",
        random_state=0,
    )
    tde.fit_predict_proba(X, y)
    tde = OrdinalTDE(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        train_estimate_method="loocv",
        random_state=0,
    )
    tde.fit_predict_proba(X, y)


def test_ordinal_tde_multivariate():
    """Test TDE with incorrect input."""
    # train TDE
    X, y = make_example_3d_numpy(n_cases=20, n_channels=10, n_timepoints=50)
    tde = IndividualOrdinalTDE(max_dims=1)
    tde._fit(X, y)
    assert len(tde._dims) == 1


def test_ordinal_tde_dict_states():
    """Test Ordinal TDE dict conversions."""
    # train TDE
    X, y = make_example_3d_numpy(n_cases=20, n_channels=10, n_timepoints=50)
    tde = IndividualOrdinalTDE(n_jobs=2)
    tde.fit(X, y)
    tde._typed_dict = True  # Example value for testing
    state = tde.__getstate__()
    assert isinstance(state, dict)
    tde.__setstate__(state)
