"""Test AutoETS."""

# __maintainer__ = []
# __all__ = []

import timeit

from statsforecast.utils import AirPassengers as ap

from aeon.forecasting._autoets import nelder_mead, optimise_params_scipy


def test_optimise_params():
    nelder_mead(ap, 2, 2, 2, 12)


def test_optimise_params_scipy():
    optimise_params_scipy(ap, 2, 2, 2, 12, method="L-BFGS-B")


def test_autoets_forecaster():
    """TestETSForecaster."""
    for _i in range(20):
        test_optimise_params()
        test_optimise_params_scipy()
    optim_ets_time = timeit.timeit(test_optimise_params, globals={}, number=1000)
    print(f"Execution time Optimise params: {optim_ets_time} seconds")  # noqa
    optim_ets_scipy_time = timeit.timeit(
        test_optimise_params_scipy, globals={}, number=1000
    )
    print(  # noqa
        f"Execution time Optimise params Scipy: {optim_ets_scipy_time} seconds"
    )


if __name__ == "__main__":
    test_autoets_forecaster()
