"""Test interval forest regressors."""

import pytest

from aeon.regression.interval_based import (
    CanonicalIntervalForestRegressor,
    DrCIFRegressor,
    RandomIntervalSpectralEnsembleRegressor,
    TimeSeriesForestRegressor,
)
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_REGRESSION
from aeon.testing.utils.estimator_checks import _assert_predict_labels
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.parametrize(
    "cls",
    [
        CanonicalIntervalForestRegressor,
        DrCIFRegressor,
        TimeSeriesForestRegressor,
        RandomIntervalSpectralEnsembleRegressor,
    ],
)
def test_tic_curves_invalid(cls):
    """Test whether temporal_importance_curves raises an error."""
    reg = cls()
    with pytest.raises(
        NotImplementedError, match="Temporal importance curves are not available."
    ):
        reg.temporal_importance_curves()


@pytest.mark.skipif(
    not _check_soft_dependencies(["pycatch22"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("cls", [CanonicalIntervalForestRegressor, DrCIFRegressor])
def test_forest_pycatch22(cls):
    """Test whether the forest regressors with pycatch22 run without error."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"]["train"]
    X_test, _ = EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"]["test"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"use_pycatch22": True})

    reg = cls(**params)
    reg.fit(X_train, y_train)
    prob = reg.predict(X_test)
    _assert_predict_labels(prob, X_test)
