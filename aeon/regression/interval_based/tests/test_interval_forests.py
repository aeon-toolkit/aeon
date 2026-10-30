"""Test interval forest regressors."""

import pytest

from aeon.regression.interval_based import (
    CanonicalIntervalForestRegressor,
    DrCIFRegressor,
    RandomIntervalSpectralEnsembleRegressor,
    TimeSeriesForestRegressor,
)


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
