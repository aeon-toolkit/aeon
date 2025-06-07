"""Function for direct forecasting."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


def direct_forecasting(forecaster, y: np.ndarray, steps_ahead: int, exog=None):
    """
    Forecast ``steps_ahead`` using a BaseForecaster instance for each horizon.

    Implements the direct strategy, cloning the given estimator and setting its horizon.

    .. code-block:: python

        preds = []
        for h in 1 to steps_ahead:
            model = clone(forecaster)
            model.horizon = h
            model.fit(y)
            preds.append(model.predict())

    Parameters
    ----------
    forecaster : BaseForecaster
        A forecasting estimator object
    y : np.ndarray
        The univariate time series to be forecast.
    steps_ahead : int
        The number of future time steps to forecast.
    exog : array-like, optional
        Optional exogenous variables to use for forecasting.

    Returns
    -------
    predictions : np.ndarray
        An array of shape (steps_ahead,) containing the forecasts for each horizon.
    """
    from copy import deepcopy

    # Check forecaster is an object and a BaseForecaster
    if isinstance(forecaster, type):
        raise TypeError(
            "Pass a forecaster instance, not a class. "
            "Use: forecaster = YourForecaster(...), not YourForecaster"
        )
    if not isinstance(forecaster, BaseForecaster):
        raise TypeError(
            "forecaster attribute must be an instance of BaseForecaster or its "
            "subclass."
        )

    preds = np.zeros(steps_ahead)
    for i in range(1, steps_ahead + 1):
        f = deepcopy(forecaster)
        f.horizon = i
        f.fit(y)
        preds[i - 1] = f.predict()
    return preds
