"""Ensemble forecaster combining multiple forecasters."""

__maintainer__ = []
__all__ = ["EnsembleForecaster"]

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class EnsembleForecaster(BaseForecaster, IterativeForecastingMixin):
    """Ensemble forecaster that combines predictions from multiple forecasters.

    Fits each component forecaster independently on the same series and combines
    their point forecasts using a specified aggregation method.  Equal weights and
    the mean are used by default; the median is recommended when robustness to a
    single poorly-fitted component is desired (as in SCUM).

    Parameters
    ----------
    forecasters : list of (str, BaseForecaster) tuples
        Named forecaster instances to include in the ensemble.  Each element is a
        ``(name, estimator)`` pair.  Names must be unique.
    weights : array-like of float or None, default=None
        Per-forecaster weights used when ``method="mean"``.  Must be non-negative
        and have the same length as ``forecasters``.  Weights are normalised to sum
        to one before use.  Ignored when ``method="median"`` or ``method`` is
        callable.  If ``None``, equal weights are used.
    method : {"mean", "median"} or callable, default="mean"
        How to combine the component forecasts at each horizon:

        - ``"mean"``   : (weighted) arithmetic mean.
        - ``"median"`` : element-wise median; ``weights`` is ignored.
        - callable     : receives an array of shape
          ``(n_forecasters, prediction_horizon)`` (or ``(n_forecasters,)`` for a
          single-step prediction) and must return a scalar or array of shape
          ``(prediction_horizon,)`` respectively.

    Attributes
    ----------
    forecasters_ : list of (str, BaseForecaster) tuples
        Fitted copies of the component forecasters.
    weights_ : np.ndarray or None
        Normalised weights used for combination, or ``None`` when ``method`` is not
        ``"mean"`` or when ``weights`` was not supplied.
    n_forecasters_ : int
        Number of component forecasters.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting import NaiveForecaster
    >>> from aeon.forecasting.ensembles import EnsembleForecaster
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> forecasters = [
    ...     ("last", NaiveForecaster(strategy="last")),
    ...     ("mean", NaiveForecaster(strategy="mean")),
    ... ]
    >>> ens = EnsembleForecaster(forecasters=forecasters, method="mean")
    >>> preds = ens.iterative_forecast(y, prediction_horizon=3)
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(self, forecasters, weights=None, method="mean"):
        self.forecasters = forecasters
        self.weights = weights
        self.method = method
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit each component forecaster on y."""
        if not callable(self.method) and self.method not in ("mean", "median"):
            raise ValueError(
                f"method must be 'mean', 'median', or a callable, got {self.method!r}"
            )

        if len(self.forecasters) == 0:
            raise ValueError("forecasters must not be empty.")

        names = [name for name, _ in self.forecasters]
        if len(names) != len(set(names)):
            raise ValueError("Forecaster names in forecasters must be unique.")

        if self.method == "mean" and self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            if w.ndim != 1:
                raise ValueError("weights must be a one-dimensional array.")
            if w.shape[0] != len(self.forecasters):
                raise ValueError(
                    f"weights has length {w.shape[0]} but there are "
                    f"{len(self.forecasters)} forecasters."
                )
            if not np.all(np.isfinite(w)):
                raise ValueError("All weights must be finite.")
            if np.any(w < 0):
                raise ValueError("All weights must be non-negative.")
            weight_sum = w.sum()
            if weight_sum <= 0:
                raise ValueError("At least one weight must be positive.")
            self.weights_ = w / weight_sum
        else:
            self.weights_ = None

        self.forecasters_ = []
        for name, forecaster in self.forecasters:
            f = _clone_estimator(forecaster)
            f.fit(y, exog)
            self.forecasters_.append((name, f))

        self.n_forecasters_ = len(self.forecasters_)
        return self

    def _predict(self, y, exog=None):
        """Return the combined one-step-ahead prediction."""
        preds = np.array([f.predict(y) for _, f in self.forecasters_])
        return float(self._combine(preds))

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        """Forecast ``prediction_horizon`` steps ahead by combining component forecasts.

        Fits all component forecasters on ``y`` once, then recursively calls each
        fitted component's ``predict`` method on an extended copy of the series.

        Parameters
        ----------
        y : np.ndarray
            The in-sample series.
        prediction_horizon : int
            Number of future steps to forecast.
        exog : np.ndarray or None, default=None
            Optional exogenous series aligned with ``y``.

        Returns
        -------
        np.ndarray
            Shape ``(prediction_horizon,)`` combined forecast.
        """
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")

        self.fit(y, exog)

        y_values = np.asarray(y, dtype=float).ravel()
        n_timepoints = y_values.shape[0]
        all_preds = np.empty((self.n_forecasters_, prediction_horizon))
        for i, (_, f) in enumerate(self.forecasters_):
            y_ext = np.empty(n_timepoints + prediction_horizon, dtype=float)
            y_ext[:n_timepoints] = y_values
            for j in range(prediction_horizon):
                pred = f.predict(y_ext[: n_timepoints + j], exog)
                all_preds[i, j] = pred
                y_ext[n_timepoints + j] = pred

        return self._combine(all_preds)

    def _combine(self, preds):
        """Combine a ``(n_forecasters, ...)`` array along axis 0."""
        if callable(self.method):
            return self.method(preds)
        if self.method == "median":
            return np.median(preds, axis=0)
        # mean, weighted or unweighted
        if self.weights_ is not None:
            return np.average(preds, weights=self.weights_, axis=0)
        return np.mean(preds, axis=0)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict
            Parameters to create a test instance of the class.
        """
        from aeon.forecasting import NaiveForecaster

        return {
            "forecasters": [
                ("last", NaiveForecaster(strategy="last")),
                ("mean", NaiveForecaster(strategy="mean")),
            ],
            "method": "mean",
        }
