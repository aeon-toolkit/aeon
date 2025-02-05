from pmdarima import auto_arima as pmd_auto_arima
from statsforecast.utils import AirPassengers as ap
from statsmodels.tsa.stattools import kpss

from aeon.forecasting._arima import auto_arima, nelder_mead
from aeon.forecasting._utils import kpss_test


def test_arima():
    model = pmd_auto_arima(
        ap,
        seasonal=True,
        m=12,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
    )
    print(model.summary())  # noqa
    print(nelder_mead(ap, 2, 1, 1, 0, 1, 0, 12))  # noqa
    print(kpss_test(ap))  # noqa
    print(kpss(ap, regression="ct", nlags=12))  # noqa
    print(auto_arima(ap))  # noqa


if __name__ == "__main__":
    test_arima()
# Fit Auto-ARIMA model
