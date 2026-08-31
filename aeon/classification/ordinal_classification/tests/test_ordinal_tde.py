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
    """Test ordinal TDE channel selection."""
    # train TDE
    X, y = make_example_3d_numpy(n_cases=20, n_channels=10, n_timepoints=50)
    tde = IndividualOrdinalTDE(max_channels=1, channel_threshold=0.0)
    tde._fit(X, y)
    assert len(tde._channels) == 1


def test_ordinal_tde_deprecated_channel_parameters_warn():
    """Test deprecated channel parameter aliases."""
    with pytest.warns(FutureWarning, match="dim_threshold"):
        individual = IndividualOrdinalTDE(dim_threshold=0.0)
    assert individual.channel_threshold == 0.0

    with pytest.warns(FutureWarning, match="max_dims"):
        individual = IndividualOrdinalTDE(max_dims=1)
    assert individual.max_channels == 1

    with pytest.warns(FutureWarning, match="dim_threshold"):
        ensemble = OrdinalTDE(dim_threshold=0.0)
    assert ensemble.channel_threshold == 0.0

    with pytest.warns(FutureWarning, match="max_dims"):
        ensemble = OrdinalTDE(max_dims=1)
    assert ensemble.max_channels == 1


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
