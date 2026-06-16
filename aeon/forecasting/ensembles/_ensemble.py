"""Ensemble forecaster combining multiple forecasters."""

__maintainer__ = []
__all__ = ["EnsembleForecaster"]

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class EnsembleForecaster(BaseForecaster, IterativeForecastingMixin):
    """Ensemble forecaster that combines predictions from multiple forecasters.

    Fits each component forecaster independently on the same series and combines
    their point forecasts using a specified averaging method. Equal weights and
    the mean are used by default; the median is recommended when robustness to a
    single poorly-fitted component is desired (as in SCUM).

    Parameters
    ----------
    forecasters : list of (str, BaseForecaster) tuples
        Named forecaster instances to include in the ensemble.  Each element is a
        ``(name, estimator)`` pair.  Names must be unique.
    weights : array-like of float or None, default=None
        Per-forecaster weights used when ``averaging_method="mean"``. Must be
        non-negative and have the same length as ``forecasters``. Weights are
        normalised to sum to one before use. Ignored when
        ``averaging_method="median"`` or ``averaging_method`` is callable. If
        ``None``, equal weights are used.
    averaging_method : {"mean", "median"} or callable, default="mean"
        How to average the component forecasts at each horizon:

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
        Normalised weights used for combination, or ``None`` when
        ``averaging_method`` is not ``"mean"`` or when ``weights`` was not supplied.
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
    >>> ens = EnsembleForecaster(forecasters=forecasters, averaging_method="mean")
    >>> preds = ens.iterative_forecast(y, prediction_horizon=3)
    """

    _tags = {
        "capability:horizon": False,
    }

    def __init__(self, forecasters, weights=None, averaging_method="mean"):
        self.forecasters = forecasters
        self.weights = weights
        self.averaging_method = averaging_method
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit each component forecaster on y."""
        self.weights_ = self._validate_parameters()
        self._clone_forecasters()
        for _, f in self.forecasters_:
            f.fit(y)

        return self

    def _predict(self, y, exog=None):
        """Return the combined one-step-ahead prediction."""
        preds = np.array([f.predict(y) for _, f in self.forecasters_])
        return float(self._combine(preds))

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        """Forecast ``prediction_horizon`` steps ahead by combining component forecasts.

        Clones each component forecaster, then asks each clone for its own full
        ``prediction_horizon`` forecast trajectory before combining those
        trajectories horizon by horizon.

        Parameters
        ----------
        y : np.ndarray
            The in-sample series.
        prediction_horizon : int
            Number of future steps to forecast.
        exog : np.ndarray or None, default=None
            Exogenous series are not currently supported by
            ``EnsembleForecaster``; non-None values are rejected during validation.

        Returns
        -------
        np.ndarray
            Shape ``(prediction_horizon,)`` combined forecast.
        """
        self._validate_prediction_horizon(prediction_horizon)

        self._preprocess_forecasting_input(y, exog, self.axis, True)
        self.weights_ = self._validate_parameters()
        self._clone_forecasters()

        component_forecasts = []
        for _, forecaster in self.forecasters_:
            preds = forecaster.iterative_forecast(
                y,
                prediction_horizon=prediction_horizon,
            )
            preds = np.asarray(preds, dtype=float).reshape(-1)
            if preds.shape[0] != prediction_horizon:
                raise ValueError(
                    "Component forecaster returned a forecast with length "
                    f"{preds.shape[0]}, expected {prediction_horizon}."
                )
            if not forecaster.is_fitted:
                forecaster.fit(y)
            component_forecasts.append(preds)

        all_preds = np.stack(component_forecasts, axis=0)
        self.is_fitted = True
        return self._combine(all_preds)

    def _combine(self, preds):
        """Combine a ``(n_forecasters, ...)`` array along axis 0."""
        if callable(self.averaging_method):
            return self.averaging_method(preds)
        if self.averaging_method == "median":
            return np.median(preds, axis=0)
        # mean, weighted or unweighted
        if self.weights_ is not None:
            return np.average(preds, weights=self.weights_, axis=0)
        return np.mean(preds, axis=0)

    def _validate_parameters(self):
        """Validate forecaster, averaging, and weight parameters."""
        if not callable(self.averaging_method) and self.averaging_method not in (
            "mean",
            "median",
        ):
            raise ValueError(
                "averaging_method must be 'mean', 'median', or a callable, "
                f"got {self.averaging_method!r}"
            )

        if len(self.forecasters) == 0:
            raise ValueError("forecasters must not be empty.")

        names = [name for name, _ in self.forecasters]
        if len(names) != len(set(names)):
            raise ValueError("Forecaster names in forecasters must be unique.")

        if self.averaging_method != "mean" or self.weights is None:
            return None

        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional array.")
        if weights.shape[0] != len(self.forecasters):
            raise ValueError(
                f"weights has length {weights.shape[0]} but there are "
                f"{len(self.forecasters)} forecasters."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("All weights must be finite.")
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative.")
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("At least one weight must be positive.")
        return weights / weight_sum

    def _validate_prediction_horizon(self, prediction_horizon):
        """Validate the iterative forecast horizon."""
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than or equal to 1.")

    def _clone_forecasters(self):
        """Clone component forecasters into ensemble state."""
        self.forecasters_ = [
            (name, _clone_estimator(forecaster))
            for name, forecaster in self.forecasters
        ]
        self.n_forecasters_ = len(self.forecasters_)

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
            "averaging_method": "mean",
        }
