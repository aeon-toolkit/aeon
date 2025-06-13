"""Test Other Packages AutoETS."""

# __maintainer__ = []
# __all__ = []

import math
import time

import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS as sktime_AutoETS
from statsforecast.models import AutoETS as sf_AutoETS
from statsforecast.utils import AirPassengers as ap
from statsforecast.utils import AirPassengersDF
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from aeon.forecasting._autoets import auto_ets
from aeon.forecasting._ets import ETSForecaster

plt.rcParams["figure.figsize"] = (12, 8)


def test_other_forecasters():
    """TestOtherForecasters."""
    plt.plot(AirPassengersDF.ds, AirPassengersDF.y, label="Actual Values", color="blue")
    # Statsmodels
    start = time.perf_counter()
    statsmodels_model = ETSModel(
        ap,
        error="mul",
        trend=None,
        damped_trend=False,
        seasonal="mul",
        seasonal_periods=12,
    )
    statsmodels_fit = statsmodels_model.fit(maxiter=10000)
    end = time.perf_counter()
    statsmodels_time = end - start
    print(  # noqa
        f"Statsmodels: Alpha: {statsmodels_fit.alpha}, \
          Beta: statsmodels_fit.beta, gamma: {statsmodels_fit.gamma}, \
            phi: statsmodels_fit.phi"
    )
    print(f"Statsmodels AIC: {statsmodels_fit.aic}")  # noqa
    sm_internal_model = ETSForecaster(
        2, 0, 2, 12, statsmodels_fit.alpha, 0, statsmodels_fit.gamma, 1
    )
    sm_internal_model.fit(ap)
    print(f"Statsmodels AIC: {sm_internal_model.aic_}")  # noqa
    plt.plot(
        AirPassengersDF.ds,
        statsmodels_fit.fittedvalues,
        label="statsmodels fit",
        color="green",
    )
    # Sktime
    start = time.perf_counter()
    sktime_model = sktime_AutoETS(auto=True, sp=12)
    sktime_model.fit(ap)
    end = time.perf_counter()
    sktime_time = end - start
    # pylint: disable=W0212
    print(  # noqa
        f"Sktime: Alpha: {sktime_model._fitted_forecaster.alpha}, \
          Beta: {sktime_model._fitted_forecaster.beta}, \
          gamma: {sktime_model._fitted_forecaster.gamma}, \
          phi: sktime_model._fitted_forecaster.phi"
    )

    if sktime_model._fitted_forecaster.error == "add":
        sk_error = 1
    elif sktime_model._fitted_forecaster.error == "mul":
        sk_error = 2
    else:
        sk_error = 0
    if sktime_model._fitted_forecaster.trend == "add":
        sk_trend = 1
    elif sktime_model._fitted_forecaster.trend == "mul":
        sk_trend = 2
    else:
        sk_trend = 0
    if sktime_model._fitted_forecaster.seasonal == "add":
        sk_seasonal = 1
    elif sktime_model._fitted_forecaster.seasonal == "mul":
        sk_seasonal = 2
    else:
        sk_seasonal = 0
    print(  # noqa
        f"Error Type: {sk_error}, Trend Type: {sk_trend}, \
          Seasonality Type: {sk_seasonal}, Seasonal Period: {12}"
    )
    print(f"Sktime AIC: {sktime_model._fitted_forecaster.aic}")  # noqa
    sk_internal_model = ETSForecaster(
        sk_error,
        sk_trend,
        sk_seasonal,
        12,
        sktime_model._fitted_forecaster.alpha,
        sktime_model._fitted_forecaster.beta,
        sktime_model._fitted_forecaster.gamma,
        1,
    )
    sk_internal_model.fit(ap)
    print(f"Sktime AIC: {sk_internal_model.aic_}")  # noqa
    plt.plot(
        AirPassengersDF.ds,
        sktime_model._fitted_forecaster.fittedvalues,
        label="sktime fitted values",
        color="red",
    )
    # pylint: enable=W0212
    # internal
    start = time.perf_counter()
    (
        error_type,
        trend_type,
        seasonality_type,
        seasonal_period,
        alpha,
        beta,
        gamma,
        phi,
    ) = auto_ets(ap)
    internal_model = ETSForecaster(
        error_type,
        trend_type,
        seasonality_type,
        seasonal_period,
        alpha,
        beta,
        gamma,
        phi,
    )
    internal_model.fit(ap)
    end = time.perf_counter()
    internal_time = end - start
    print(  # noqa
        f"Internal: Alpha: {internal_model.alpha}, Beta: {internal_model.beta}, \
          gamma: {internal_model.gamma}, phi: {internal_model.phi}"
    )
    print(  # noqa
        f"Error Type: {internal_model.error_type}, \
          Trend Type: {internal_model.trend_type}, \
          Seasonality Type: {internal_model.seasonality_type}, \
            Seasonal Period: {internal_model.seasonal_period}"
    )
    print(f"Internal AIC: {internal_model.aic_}")  # noqa
    plt.plot(
        AirPassengersDF.ds[seasonal_period:],
        internal_model.fitted_values_,
        label="Internal fitted values",
        color="black",
    )
    # statsforecast
    start = time.perf_counter()
    sf_model = sf_AutoETS(season_length=12)
    sf_model.fit(ap)
    end = time.perf_counter()
    statsforecast_time = end - start
    print(  # noqa
        f"Statsforecast: Alpha: {sf_model.model_['par'][0]}, \
            Beta: {sf_model.model_['par'][1]}, gamma: {sf_model.model_['par'][2]}, \
                phi: {sf_model.model_['par'][3]}"
    )
    print(  # noqa
        f"Statsforecast Model Type: {sf_model.model_['method']}, \
          AIC: {sf_model.model_['aic']}"
    )
    sf_internal_model = ETSForecaster(
        2 if sf_model.model_["components"][0] == "M" else 1,
        (
            2
            if sf_model.model_["components"][1] == "M"
            else 1 if sf_model.model_["components"][1] == "A" else 0
        ),
        (
            2
            if sf_model.model_["components"][2] == "M"
            else 1 if sf_model.model_["components"][2] == "A" else 0
        ),
        12,
        0 if math.isnan(sf_model.model_["par"][0]) else sf_model.model_["par"][0],
        0 if math.isnan(sf_model.model_["par"][1]) else sf_model.model_["par"][1],
        0 if math.isnan(sf_model.model_["par"][2]) else sf_model.model_["par"][2],
        0 if math.isnan(sf_model.model_["par"][3]) else sf_model.model_["par"][3],
    )
    sf_internal_model.fit(ap)
    print(f"Statsforecast AIC: {sf_internal_model.aic_}")  # noqa
    plt.plot(
        AirPassengersDF.ds,
        sf_model.model_["fitted"],
        label="statsforecast fitted values",
        color="orange",
    )
    print(  # noqa
        f"Statsmodels Time: {statsmodels_time}\
            Sktime Time: {sktime_time}\
            Internal Time: {internal_time}\
            Statsforecast Time: {statsforecast_time}"
    )  # noqa
    plt.ylabel("Air Passenger Numbers")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_other_forecasters()
