"""Stats based forecasters."""

__all__ = [
    "ETS",
    "ARIMA",
]

from aeon.forecasting.stats._arima import ARIMA
from aeon.forecasting.stats._ets import ETS
