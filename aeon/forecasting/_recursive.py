"""Function for direct forecasting."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


def recursive_forecasting(
    forecaster: BaseForecaster, y: np.ndarray, steps_ahead: int, exog=None
):
    """
    Forecast ``steps_ahead`` using a BaseForecaster trained on y then recursively .

    This function implements the "direct" forecasting strategy, where a separate
    forecasting model is trained for each horizon from 1 to ``steps_ahead``.

    The core logic is:

    .. code-block:: python

        model = BaseForecaster(horizon=1)
        model.fit(y, exog)
        for i in range(1, steps_ahead + 1):
            predictions[i] = model.predict(y)
            y.append(prediction[i])

    Parameters
    ----------
    forecaster : BaseForecaster object
        A forecasting estimator object.
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
    >>> from aeon.forecasting import recursive_forecasting
    >>> from aeon.forecasting import RegressionForecaster
    >>> import numpy as np
    >>> y = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> forecaster=RegressionForecaster(horizon=1, window=3)
    >>> pred = recursive_forecasting(forecaster,y,steps_ahead=5)
    >>> len(pred)
    5
    >>> pred
    array([11., 12., 13., 14., 15.])
    """
    if isinstance(forecaster, type):
        raise TypeError(
            "Passed a class in the forecaster parameter rather than a "
            "object that is an instance of class that inherits from BaseForecaster"
        )
    if not isinstance(forecaster, BaseForecaster):
        raise TypeError("Passed an object that does not inherit from BaseForecaster.")
    preds = np.zeros(steps_ahead)
    forecaster.fit(y, exog=exog)
    for i in range(1, steps_ahead + 1):
        preds[i - 1] = forecaster.predict(y, exog)
        y = np.append(y, preds[i - 1])
    return preds
