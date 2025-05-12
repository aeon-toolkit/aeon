from pmdarima import auto_arima as pmd_auto_arima
from statsforecast.utils import AirPassengers as ap
from statsmodels.tsa.stattools import kpss

from aeon.forecasting._arima import ARIMAForecaster, auto_arima, nelder_mead
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
    print(f"Optimal Model: {nelder_mead(ap, 2, 1, 1, 0, 1, 0, 12, True)}")  # noqa
    print(model.predict(n_periods=1))  # noqa
    print(kpss_test(ap))  # noqa
    print(kpss(ap, regression="c", nlags=12))  # noqa
    print(auto_arima(ap))  # noqa
    forecaster = ARIMAForecaster()
    forecaster.fit(ap)
    print(forecaster.predict())  # noqa


if __name__ == "__main__":
    test_arima()
# Fit Auto-ARIMA model
