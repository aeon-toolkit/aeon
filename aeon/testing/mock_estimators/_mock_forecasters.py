"""Mock forecasters useful for testing and debugging.

Used in the forecasting module to test composites and pipelines.
"""

__maintainer__ = []

__all__ = ["MockForecaster", "MockUnivariateForecasterLogger"]


import re
from copy import deepcopy
from functools import wraps
from inspect import getcallargs, getfullargspec

import pandas as pd

from aeon.base import BaseEstimator
from aeon.forecasting.base import BaseForecaster


class _MockEstimatorMixin:
    """Mixin class for constructing mock estimators."""

    @property
    def log(self):
        """Log of the methods called and the parameters passed in each method."""
        if not hasattr(self, "_MockEstimatorMixin__log"):
            return []
        else:
            return self._MockEstimatorMixin__log

    def add_log_item(self, value):
        """Append an item to the log.

        State change:
        self.log - `value` is appended to the list self.log

        Parameters
        ----------
        value : any object
        """
        if not hasattr(self, "_MockEstimatorMixin__log"):
            self._MockEstimatorMixin__log = [value]
        else:
            self._MockEstimatorMixin__log = self._MockEstimatorMixin__log + [value]


def _method_logger(method):
    """Log the method and it's arguments."""

    @wraps(wrapped=method)
    def wrapper(self, *args, **kwargs):
        args_dict = getcallargs(method, self, *args, **kwargs)
        if not isinstance(self, _MockEstimatorMixin):
            raise TypeError("method_logger requires a MockEstimator class")
        args_dict.pop("self")
        self.add_log_item((method.__name__, deepcopy(args_dict)))
        return method(self, *args, **kwargs)

    return wrapper


def make_mock_estimator(
    estimator_class: BaseEstimator, method_regex: str = ".*"
) -> BaseEstimator:
    r"""Transform any estimator class into a mock estimator class.

    The returned class will accept the original arguments passed in estimator_class
    __init__ as a dictionary of kwargs.

    Parameters
    ----------
    estimator_class : BaseEstimator
        any aeon estimator
    method_regex : str, optional
        regex to filter methods on, by default ".*"
        Useful regex examples:
            - everything: '.*'
            - private methods only: '^(?!^__\w+__$)^_\w'
            - public methods only: '(?!^_\w+)'

    Returns
    -------
    BaseEstimator
        input estimator class with logging feature enabled

    Examples
    --------
    >>> from aeon.forecasting.naive import NaiveForecaster
    >>> from aeon.testing.mock_estimators import make_mock_estimator
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> mock_estimator_class = make_mock_estimator(NaiveForecaster)
    >>> mock_estimator_instance = mock_estimator_class({"strategy": "last", "sp": 1})
    >>> mock_estimator_instance.fit(y)
    _MockEstimator(...)

    """
    dunder_methods_regex = r"^__\w+__$"

    class _MockEstimator(estimator_class, _MockEstimatorMixin):
        def __init__(self, estimator_kwargs=None):
            self.estimator_kwargs = estimator_kwargs
            if estimator_kwargs is not None:
                super().__init__(**estimator_kwargs)
            else:
                super().__init__()

    for attr_name in dir(estimator_class):
        attr = getattr(_MockEstimator, attr_name)
        # exclude dunder methods (e.g. __eq__, __class__ etc.) and non callables
        # from logging
        if not re.match(dunder_methods_regex, attr_name) and callable(attr):
            # match the given regex pattern
            # exclude static and class methods from logging
            if (
                re.match(method_regex, attr_name)
                and "self" in getfullargspec(attr).args
            ):
                setattr(_MockEstimator, attr_name, _method_logger(attr))

    return _MockEstimator


class MockUnivariateForecasterLogger(BaseForecaster, _MockEstimatorMixin):
    """Mock univariate forecaster that logs the methods called and their parameters.

    Parameters
    ----------
    prediction_constant : float, optional
        The forecasted value for all steps in the horizon, by default 10

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> from aeon.testing.mock_estimators import MockUnivariateForecasterLogger
    >>> y = load_airline()
    >>> forecaster = MockUnivariateForecasterLogger()
    >>> forecaster.fit(y)
    MockUnivariateForecasterLogger(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    >>> print(forecaster.log)
    [('_fit', ...), ('_predict', ...)]
    """

    _tags = {
        "y_input_type": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_type": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_type": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
    }

    def __init__(self, prediction_constant: float = 10):
        self.prediction_constant = prediction_constant
        super().__init__()

    @_method_logger
    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_type")
            Time series to which to fit the forecaster.
            if self.get_tag("y_input_type")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("y_input_type")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("y_input_type")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_type")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        return self

    @_method_logger
    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        index = fh.to_absolute(self.cutoff).to_pandas()
        return pd.Series(self.prediction_constant, index=index)

    @_method_logger
    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_type")
            Time series with which to update the forecaster.
            if self.get_tag("y_input_type")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("y_input_type")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("y_input_type")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        return self

    @_method_logger
    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        fh_index = fh.to_absolute(self.cutoff).to_pandas()
        col_index = pd.MultiIndex.from_product([["Quantiles"], alpha])
        pred_quantiles = pd.DataFrame(columns=col_index, index=fh_index)

        for a in alpha:
            pred_quantiles[("Quantiles", a)] = pd.Series(
                self.prediction_constant * 2 * a, index=fh_index
            )

        return pred_quantiles


class MockForecaster(BaseForecaster):
    """Mock forecaster, returns data frames filled with a constant.

    Parameters
    ----------
    prediction_constant : float, optional
        The forecasted value for all steps in the horizon, by default 10

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> from aeon.testing.mock_estimators import MockUnivariateForecasterLogger
    >>> y = load_airline()
    >>> forecaster = MockUnivariateForecasterLogger()
    >>> forecaster.fit(y)
    MockUnivariateForecasterLogger(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    >>> print(forecaster.log)
    [('_fit', ...), ('_predict', ...)]
    """

    _tags = {
        "y_input_type": "both",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_type": "pd.DataFrame",  # which types do _fit, _predict, assume for y?
        "X_inner_type": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
    }

    def __init__(self, prediction_constant: float = 10):
        self.prediction_constant = prediction_constant
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_type")
            Time series to which to fit the forecaster.
            if self.get_tag("y_input_type")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("y_input_type")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("y_input_type")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_type")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        index = fh.to_absolute(self.cutoff).to_pandas()
        return pd.DataFrame(
            self.prediction_constant, index=index, columns=self._y.columns
        )

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_type")
            Time series with which to update the forecaster.
            if self.get_tag("y_input_type")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("y_input_type")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("y_input_type")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        return self

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        cols = self._y.columns

        if len(cols) == 1:
            cols = ["Quantiles"]

        col_index = pd.MultiIndex.from_product([cols, alpha])
        fh_index = fh.to_absolute(self.cutoff).to_pandas()
        pred_quantiles = pd.DataFrame(index=fh_index, columns=col_index)

        for col, a in col_index:
            pred_quantiles[col, a] = pd.Series(
                self.prediction_constant * 2 * a, index=fh_index
            )

        return pred_quantiles

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [{"prediction_constant": 42}, {"prediction_constant": -4.2}]
