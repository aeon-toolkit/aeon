"""Tests for Bagging Forecasters."""

import pandas as pd
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.compose import BaggingForecaster
from aeon.forecasting.compose._bagging import _calculate_data_quantiles
from aeon.forecasting.naive import NaiveForecaster
from aeon.transformations.bootstrap import STLBootstrapTransformer
from aeon.transformations.boxcox import LogTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies

y = load_airline()


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize("transformer", [LogTransformer, NaiveForecaster])
def test_bagging_forecaster_transformer_type_error(transformer):
    """Test that the right exception is raised for invalid transformer."""
    with pytest.raises(TypeError) as ex:
        f = BaggingForecaster(
            bootstrap_transformer=transformer, forecaster=NaiveForecaster(sp=12)
        )
        f.fit(y)
        msg = (
            "bootstrap_transformer in BaggingForecaster should be a Transformer "
            "that take as input a Series and output a Panel."
        )
        assert msg == ex.value


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize("forecaster", [LogTransformer])
def test_bagging_forecaster_forecaster_type_error(forecaster):
    """Test that the right exception is raised for invalid forecaster."""
    with pytest.raises(TypeError) as ex:
        f = BaggingForecaster(
            bootstrap_transformer=STLBootstrapTransformer(sp=12),
            forecaster=forecaster,
        )
        f.fit(y)
        msg = "forecaster in BaggingForecaster should be an aeon Forecaster"
        assert msg == ex.value


def test_calculate_data_quantiles():
    """Test that we calculate quantiles correctly."""
    series_names = ["s1", "s2", "s3"]
    fh = [1, 2]
    alpha = [0, 0.5, 1]
    quantiles_column_index = pd.MultiIndex.from_product([["Quantiles"], alpha])

    index = pd.MultiIndex.from_product(
        [series_names, fh], names=["time_series", "time"]
    )
    df = pd.DataFrame(data=[1, 10, 2, 11, 3, 12], index=index)

    output_df = pd.DataFrame(
        data=[[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
        columns=quantiles_column_index,
        index=pd.Index(data=fh, name="time"),
    )

    pd.testing.assert_frame_equal(_calculate_data_quantiles(df, alpha), output_df)
