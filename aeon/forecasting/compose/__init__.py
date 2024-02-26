"""Implements composite forecasters."""

__all__ = [
    "HierarchyEnsembleForecaster",
    "ColumnEnsembleForecaster",
    "EnsembleForecaster",
    "AutoEnsembleForecaster",
    "TransformedTargetForecaster",
    "ForecastingPipeline",
    "ForecastX",
    "DirectTabularRegressionForecaster",
    "DirectTimeSeriesRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "DirRecTabularRegressionForecaster",
    "DirRecTimeSeriesRegressionForecaster",
    "StackingForecaster",
    "MultiplexForecaster",
    "make_reduction",
    "BaggingForecaster",
    "ForecastByLevel",
    "Permute",
]

from aeon.forecasting.compose._bagging import BaggingForecaster
from aeon.forecasting.compose._column_ensemble import ColumnEnsembleForecaster
from aeon.forecasting.compose._ensemble import (
    AutoEnsembleForecaster,
    EnsembleForecaster,
)
from aeon.forecasting.compose._grouped import ForecastByLevel
from aeon.forecasting.compose._hierarchy_ensemble import HierarchyEnsembleForecaster
from aeon.forecasting.compose._multiplexer import MultiplexForecaster
from aeon.forecasting.compose._pipeline import (
    ForecastingPipeline,
    ForecastX,
    Permute,
    TransformedTargetForecaster,
)
from aeon.forecasting.compose._reduce import (
    DirectTabularRegressionForecaster,
    DirectTimeSeriesRegressionForecaster,
    DirRecTabularRegressionForecaster,
    DirRecTimeSeriesRegressionForecaster,
    MultioutputTabularRegressionForecaster,
    MultioutputTimeSeriesRegressionForecaster,
    RecursiveTabularRegressionForecaster,
    RecursiveTimeSeriesRegressionForecaster,
    make_reduction,
)
from aeon.forecasting.compose._stack import StackingForecaster
