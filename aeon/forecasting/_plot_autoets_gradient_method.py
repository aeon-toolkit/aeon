"""Test AutoETS."""

# __maintainer__ = []
# __all__ = []

import matplotlib.pyplot as plt
from statsforecast.utils import AirPassengers as ap
from statsforecast.utils import AirPassengersDF

from aeon.forecasting._autoets import auto_ets
from aeon.forecasting._ets import ETSForecaster

plt.rcParams["figure.figsize"] = (12, 8)


def test_autoets_forecaster():
    """TestETSForecaster."""
    (
        error_type,
        trend_type,
        seasonality_type,
        seasonal_period,
        alpha,
        beta,
        gamma,
        phi,
    ) = auto_ets(ap, method="internal_gradient")
    print(  # noqa
        f"Error Type: {error_type}, Trend Type: {trend_type}, \
        Seasonality Type: {seasonality_type}, Seasonal Period: {seasonal_period}, \
        Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Phi: {phi}"
    )  # noqa
    etsforecaster = ETSForecaster(
        error_type,
        trend_type,
        seasonality_type,
        seasonal_period,
        alpha,
        beta,
        gamma,
        phi,
        1,
    )
    etsforecaster.fit(ap)
    print(f"liklihood: {etsforecaster.liklihood_}")  # noqa

    # assert np.allclose([parameter.item() for parameter in parameters],
    # [0.1,0.05,0.05,0.98])
    plt.plot(AirPassengersDF.ds, AirPassengersDF.y, label="Actual Values", color="blue")
    plt.plot(
        AirPassengersDF.ds,
        etsforecaster.fitted_values_,
        label="Predicted Values",
        color="green",
    )
    plt.plot(
        AirPassengersDF.ds, etsforecaster.residuals_, label="Residuals", color="red"
    )
    plt.ylabel("Air Passenger Numbers")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_autoets_forecaster()
