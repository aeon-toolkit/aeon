"""Function for direct forecasting."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


def direct_forecasting(forecaster_class, y: np.ndarray, steps_ahead: int, exog=None):
    """
    Forecast ``steps_ahead`` using a BaseForecaster trained on different horizons.

    This function implements the "direct" forecasting strategy, where a separate
    forecasting model is trained for each horizon from 1 to ``steps_ahead``.

    The core logic is:

    .. code-block:: python

        for i in range(1, steps_ahead + 1):
            model = BaseForecaster(horizon=i)
            model.fit(y, exog)
            predictions[i] = model.predict()

    Parameters
    ----------
    forecaster_class : BaseForecaster
        A forecasting estimator class that supports setting the forecast horizon at
        instantiation.
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

    Example
    -------
    >>> from aeon.forecasting import direct_forecasting, ETSForecaster
    >>> import numpy as np
    >>> y = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> pred = direct_forecasting(ETSForecaster,y,steps_ahead=5)
    >>> len(pred)
    5
    >>> pred
    array([4.4867844, 4.4867844, 4.4867844, 4.4867844, 4.4867844])
    """
    if not isinstance(forecaster_class, type):
        raise TypeError(
            "Passed an object in the forecaster_class parameter rather than a "
            "class that inherits from BaseForecaster"
        )
    if not issubclass(forecaster_class, BaseForecaster):
        raise TypeError(
            "Passed a class in the forecaster_class parameter that does not "
            "extend BaseForecaster. It must inherit from BaseForecaster."
        )
    preds = np.zeros(steps_ahead)
    for i in range(1, steps_ahead + 1):
        f = forecaster_class(horizon=i)
        f.fit(y, exog=exog)
        preds[i - 1] = f.forecast(y, exog)
    return preds
