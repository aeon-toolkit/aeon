"""Tests for N-BEATS Forecaster."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.deep_learning._nbeats import NBeatsForecaster
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("horizon,window,epochs", [(1, 10, 2), (2, 12, 1)])
def test_nbeats_forecaster_fit_predict(horizon, window, epochs):
    """Test NBeatsForecaster fit and predict with different parameter combinations."""
    import tensorflow as tf

    y = load_airline()

    forecaster = NBeatsForecaster(
        horizon=horizon,
        window=window,
        n_epochs=epochs,
        batch_size=4,
        steps_per_epoch=2,
        verbose=0,
    )

    forecaster.fit(y)
    prediction = forecaster.predict(y)

    assert prediction is not None
    if horizon == 1:
        assert np.ndim(prediction) <= 1
    else:
        assert len(prediction) == horizon

    if isinstance(prediction, tf.Tensor):
        assert not tf.math.is_nan(prediction).numpy().any()
    else:
        assert not np.isnan(prediction).any()


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_stack_configurations():
    """Test specific N-BEATS stack configurations (Generic vs Interpretable)."""
    y = load_airline()[:50]

    model_generic = NBeatsForecaster(
        window=10,
        horizon=2,
        stack_types=["generic"],
        nb_blocks_per_stack=1,
        n_epochs=1,
        steps_per_epoch=1,
        verbose=0,
    )
    model_generic.fit(y)
    pred_generic = model_generic.predict(y)
    assert len(pred_generic) == 2

    model_interp = NBeatsForecaster(
        window=10,
        horizon=2,
        stack_types=["trend", "seasonality"],
        nb_blocks_per_stack=1,
        trend_degree=2,
        num_harmonics=2,
        n_epochs=1,
        steps_per_epoch=1,
        verbose=0,
    )
    model_interp.fit(y)
    pred_interp = model_interp.predict(y)
    assert len(pred_interp) == 2


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_decomposition():
    """Test the interpretability feature (predict_decomposition)."""
    y = load_airline()[:60]

    forecaster = NBeatsForecaster(
        window=15,
        horizon=4,
        stack_types=["trend", "seasonality"],
        nb_blocks_per_stack=1,
        n_epochs=1,
        steps_per_epoch=1,
        verbose=0,
    )

    forecaster.fit(y)
    decomposition = forecaster.predict_decomposition(y)

    assert isinstance(decomposition, dict)
    assert "total_forecast" in decomposition

    keys = list(decomposition.keys())
    assert any("trend" in k for k in keys)
    assert any("seasonality" in k for k in keys)

    assert decomposition["total_forecast"].shape == (4,)

    component_sum = np.zeros_like(decomposition["total_forecast"])
    for k, v in decomposition.items():
        if k != "total_forecast":
            component_sum += v

    np.testing.assert_allclose(
        component_sum,
        decomposition["total_forecast"],
        rtol=1e-5,
        err_msg="Sum of N-BEATS components does not equal total forecast",
    )


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_nbeats_custom_loss_and_sharing():
    """Test compilation with SMAPE loss and weight sharing enabled."""
    y = load_airline()[:40]

    forecaster = NBeatsForecaster(
        window=10,
        horizon=1,
        stack_types=["generic", "generic"],
        nb_blocks_per_stack=2,
        share_weights_in_stack=True,
        loss="smape",
        n_epochs=1,
        steps_per_epoch=1,
        verbose=0,
    )

    forecaster.fit(y)
    assert forecaster.model_ is not None
