"""Tests for Deseasonalizer."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.forecasting.model_selection import temporal_train_test_split
from aeon.forecasting.tests import TEST_SPS
from aeon.testing.utils.data_gen import make_forecasting_problem
from aeon.transformations.detrend import Deseasonalizer
from aeon.utils.validation._dependencies import _check_soft_dependencies

MODELS = ["additive", "multiplicative"]

y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("sp", TEST_SPS)
def test_deseasonalised_values(sp):
    from statsmodels.tsa.seasonal import seasonal_decompose

    transformer = Deseasonalizer(sp=sp)
    transformer.fit(y_train)
    actual = transformer.transform(y_train)

    r = seasonal_decompose(y_train, period=sp)
    expected = y_train - r.seasonal
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("model", MODELS)
def test_transform_time_index(sp, model):
    transformer = Deseasonalizer(sp=sp, model=model)
    transformer.fit(y_train)
    yt = transformer.transform(y_test)
    np.testing.assert_array_equal(yt.index, y_test.index)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("model", MODELS)
def test_inverse_transform_time_index(sp, model):
    transformer = Deseasonalizer(sp=sp, model=model)
    transformer.fit(y_train)
    yit = transformer.inverse_transform(y_test)
    np.testing.assert_array_equal(yit.index, y_test.index)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("model", MODELS)
def test_transform_inverse_transform_equivalence(sp, model):
    transformer = Deseasonalizer(sp=sp, model=model)
    transformer.fit(y_train)
    yit = transformer.inverse_transform(transformer.transform(y_train))
    np.testing.assert_array_equal(y_train.index, yit.index)
    np.testing.assert_array_almost_equal(y_train, yit)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_deseasonalizer_in_pipeline():
    """Test deseasonalizer in pipeline, see issue #3267."""
    from aeon.datasets import load_airline
    from aeon.forecasting.compose import TransformedTargetForecaster
    from aeon.forecasting.theta import ThetaForecaster
    from aeon.transformations.detrend import Deseasonalizer

    all_df = load_airline().to_frame()

    model = TransformedTargetForecaster(
        [
            ("deseasonalize", Deseasonalizer(model="additive", sp=12)),
            ("forecast", ThetaForecaster()),
        ]
    )
    train_df = all_df["1949":"1950"]
    model.fit(train_df)
    model.update(y=all_df.loc["1951"])
