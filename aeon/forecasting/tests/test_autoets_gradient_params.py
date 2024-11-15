"""Test AutoETS."""

# __maintainer__ = []
# __all__ = []
# import numpy as np

from statsforecast.utils import AirPassengers as ap

from aeon.forecasting._autoets_gradient_params import _fit


def test_autoets_forecaster():
    """TestETSForecaster."""
    parameters = _fit(ap, 1, 1, 1, 12)
    print(parameters)  # noqa
    # assert np.allclose([parameter.item() for parameter in parameters],
    # [0.1,0.05,0.05,0.98])


if __name__ == "__main__":
    test_autoets_forecaster()
