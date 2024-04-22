"""Tests for using OptionalPassthrough."""

import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_optionalpassthrough():
    """Test for OptionalPassthrough used within grid search and with pipeline.

    Same as docstring example of OptionalPassthrough.
    """
    from sklearn.preprocessing import StandardScaler

    from aeon.datasets import load_airline
    from aeon.forecasting.compose import TransformedTargetForecaster
    from aeon.forecasting.model_selection import (
        ForecastingGridSearchCV,
        SlidingWindowSplitter,
    )
    from aeon.forecasting.naive import NaiveForecaster
    from aeon.transformations.adapt import TabularToSeriesAdaptor
    from aeon.transformations.compose import OptionalPassthrough
    from aeon.transformations.detrend import Deseasonalizer

    # create pipeline
    pipe = TransformedTargetForecaster(
        steps=[
            ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
            ("scaler", OptionalPassthrough(TabularToSeriesAdaptor(StandardScaler()))),
            ("forecaster", NaiveForecaster()),
        ]
    )
    # putting it all together in a grid search
    cv = SlidingWindowSplitter(
        initial_window=60, window_length=24, start_with_window=True, step_length=48
    )
    param_grid = {
        "deseasonalizer__passthrough": [True, False],
        "scaler__transformer__transformer__with_mean": [True, False],
        "scaler__passthrough": [True, False],
        "forecaster__strategy": ["drift", "mean", "last"],
    }
    gscv = ForecastingGridSearchCV(
        forecaster=pipe, param_grid=param_grid, cv=cv, n_jobs=-1
    )
    gscv.fit(load_airline())


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_passthrough_does_not_broadcast_variables():
    """Test that OptionalPassthrough does not itself vectorize/broadcast columns."""
    from aeon.datasets import load_longley
    from aeon.transformations.compose import OptionalPassthrough
    from aeon.transformations.detrend import Deseasonalizer

    _, X = load_longley()

    t = OptionalPassthrough(Deseasonalizer())
    t.fit(X)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_passthrough_does_not_broadcast_instances():
    """Test that OptionalPassthrough does not itself vectorize/broadcast rows."""
    from aeon.testing.utils.data_gen import _make_hierarchical
    from aeon.transformations.compose import OptionalPassthrough
    from aeon.transformations.detrend import Deseasonalizer

    X = _make_hierarchical()

    t = OptionalPassthrough(Deseasonalizer())
    t.fit(X)
