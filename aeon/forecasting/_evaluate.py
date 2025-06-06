"""Evaluate forecaster."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


def evaluate_forecaster(
    forecaster: BaseForecaster, y_train: np.ndarray, y_test: np.ndarray
):
    """
    Forecast on increasng train data.

    This function evaluates a forecaster with a single step horizon on an increasing
    length time series. The initial train series and test series are passed as
    arguments. They are assumed to be contiguous, i.e. last element in train is the
    observation one before the first in test. The forecaster is trained on
    the initial train set and predicts the next value. The first test value is then
    appended to the test, and the training is repeated.

    Does not yet accept exogenous variable.

    The core logic is:

    .. code-block:: python

        model = BaseForecaster(horizon=1)
        for i in range(1, len(y_test)):
            model.fit(y_train, exog)
            predictions[i] = model.predict()
            y_train.append(y_test[i])

    Parameters
    ----------
    forecaster : BaseForecaster object
        A forecasting estimator object.
    y_train : np.ndarray
        The univariate initial training time series.
    y_test : np.ndarray
        The test series to make predictions for.

    Returns
    -------
    predictions : np.ndarray
        An array of shape (len(train_y-1,) containing the forecasts for each horizon.

    Example
    -------
    >>> from aeon.forecasting import evaluate_forecaster
    >>> from aeon.forecasting import RegressionForecaster
    >>> import numpy as np
    >>> y_train = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> y_test = np.array([11,12,13,14,15])
    >>> forecaster=RegressionForecaster(horizon=1, window=3)
    >>> pred = evaluate_forecaster(forecaster,y_train, y_test)
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
    preds = np.zeros(len(y_test))
    for i in range(0, len(y_test)):
        forecaster.fit(y_train)
        preds[i] = forecaster.predict()
        y_train = np.append(y_train, y_test[i])
    return preds
