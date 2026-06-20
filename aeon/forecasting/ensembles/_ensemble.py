"""Ensemble forecaster combining multiple forecasters."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["EnsembleForecaster"]

import numpy as np

from aeon.base import ComposableEstimatorMixin
from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class EnsembleForecaster(
    ComposableEstimatorMixin, BaseForecaster, IterativeForecastingMixin
):
    """Ensemble forecaster that combines predictions from multiple forecasters.

    Fits each component forecaster independently on the same series and combines
    their point forecasts using a specified averaging method. Equal weights and
    the mean are used by default; the median is recommended when robustness to a
    single poorly-fitted component is desired (as in SCUM).

    Parameters
    ----------
    forecasters : list of BaseForecaster or list of (str, BaseForecaster) tuples
        Forecaster instances to include in the ensemble. Each element can be a
        forecaster instance or a ``(name, estimator)`` pair. Names must be unique
        when supplied.
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
    iterative_strategy : {"component", "ensemble"}, default="component"
        Strategy used by ``iterative_forecast``. If ``"component"``, each
        component forecaster is iterated forward independently using its own
        forecasts and the resulting forecast paths are aggregated at each horizon. If
        ``"ensemble"``, the ensemble prediction at each step is appended to a shared
        history and passed to each fitted component's ``predict`` method. The
        default ``"component"`` strategy is the safer general-purpose choice for
        stateful forecasters, because it delegates multi-step recursion to each
        component.

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
    >>> preds.shape
    (3,)

    See Also
    --------
    aeon.forecasting.stats.SCUM : Simple Combination of Univariate Models forecaster.

    References
    ----------
    .. [1] Petropoulos, F. and Svetunkov, I. (2020). A simple combination of
       univariate models. International Journal of Forecasting, 36(1), 110-115.
    """

    _tags = {
        "capability:horizon": False,
    }
    _estimators_attr = "_forecasters"
    _fitted_estimators_attr = "forecasters_"

    def __init__(
        self,
        forecasters,
        weights=None,
        averaging_method="mean",
        iterative_strategy="component",
    ):
        self.forecasters = forecasters
        self.weights = weights
        self.averaging_method = averaging_method
        self.iterative_strategy = iterative_strategy
        BaseForecaster.__init__(self, horizon=1, axis=1)

        self._normalise_forecasters()

    def set_params(self, **params):
        """Set estimator parameters, including nested forecaster parameters."""
        sentinel = object()
        forecasters = params.pop("forecasters", sentinel)
        if forecasters is not sentinel:
            self.forecasters = forecasters
            self._normalise_forecasters()

        forecaster_names = {name for name, _ in self._forecasters}
        direct_replacements = [
            name for name in params if "__" not in name and name in forecaster_names
        ]
        super().set_params(**params)

        # The mixin replaces entries in ``_forecasters`` for direct component
        # assignments. Keep the public constructor parameter in sync for clone().
        if direct_replacements:
            self.forecasters = list(self._forecasters)
            self._normalise_forecasters()
        return self

    def _normalise_forecasters(self):
        """Validate and convert component forecasters to named tuples."""
        if isinstance(self.forecasters, list) and len(self.forecasters) == 0:
            raise ValueError("forecasters must not be empty.")
        self._check_estimators(
            self.forecasters,
            attr_name="forecasters",
            class_type=BaseForecaster,
        )
        self._forecasters = self._convert_estimators(
            self.forecasters, clone_estimators=False
        )

    def _fit(self, y, exog=None):
        """Fit each component forecaster on y."""
        self._setup_components()
        for _, f in self.forecasters_:
            f.fit(y)

        return self

    def _predict(self, y, exog=None):
        """Return the combined one-step-ahead prediction."""
        preds = np.array([f.predict(y) for _, f in self.forecasters_])
        return self._combine(preds)

    def iterative_forecast(
        self,
        y,
        prediction_horizon,
        exog=None,
        *,
        future_exog=None,
    ):
        """Forecast ``prediction_horizon`` steps ahead by combining component forecasts.

        The ``iterative_strategy`` parameter controls the recursive semantics.
        With ``"component"``, each component forecaster produces its own full
        forecast path, then paths are aggregated horizon by horizon. With
        ``"ensemble"``, the ensemble produces one aggregated prediction at a
        time and appends that prediction to a shared history passed to
        ``predict``. Components are fitted once on the original history; whether
        the extended context affects later steps depends on the component's
        ``predict`` implementation.

        Parameters
        ----------
        y : np.ndarray
            The in-sample series.
        prediction_horizon : int
            Number of future steps to forecast.
        exog : np.ndarray or None, default=None
            Exogenous series are not currently supported by
            ``EnsembleForecaster``; non-None values are rejected during validation.
        future_exog : np.ndarray or None, default=None
            Future exogenous values are not currently supported by
            ``EnsembleForecaster``; non-None values are rejected during validation.

        Returns
        -------
        np.ndarray
            Shape ``(prediction_horizon,)`` combined forecast.
        """
        y, _, _ = self._check_iterative_forecast_inputs(y, prediction_horizon)
        prediction_horizon = int(prediction_horizon)
        if exog is not None or future_exog is not None:
            raise ValueError(
                f"Exogenous variables passed but {self.__class__.__name__} cannot "
                "handle exogenous variables"
            )

        y, exog = self._preprocess_forecasting_input(y, None, self.axis, True)
        self._setup_components()

        if self.iterative_strategy == "component":
            predictions = self._component_iterative_forecast(y, prediction_horizon)
        else:
            predictions = self._ensemble_iterative_forecast(y, prediction_horizon)
        self.is_fitted = True
        return predictions

    def _setup_components(self):
        """Validate parameters and initialise fitted component clones."""
        self.weights_ = self._validate_parameters()
        self.forecasters_ = self._convert_estimators(self._forecasters)
        self.n_forecasters_ = len(self.forecasters_)

    def _component_iterative_forecast(self, y, prediction_horizon):
        """Iterate forecasts for each component path, then combine paths."""
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
            component_forecasts.append(preds)

        all_preds = np.stack(component_forecasts, axis=0)
        return self._combine(all_preds)

    def _ensemble_iterative_forecast(self, y, prediction_horizon):
        """Iterate forecasts using the combined ensemble prediction as feedback."""
        for _, forecaster in self.forecasters_:
            forecaster.fit(y)

        y_extended = np.asarray(y, dtype=float).reshape(-1)
        predictions = np.zeros(prediction_horizon, dtype=float)
        for i in range(prediction_horizon):
            component_preds = np.array(
                [forecaster.predict(y_extended) for _, forecaster in self.forecasters_],
                dtype=float,
            )
            ensemble_pred = self._combine(component_preds)
            predictions[i] = ensemble_pred
            y_extended = np.append(y_extended, ensemble_pred)
        return predictions

    def _combine(self, preds):
        """Combine a ``(n_forecasters, ...)`` array along axis 0."""
        if callable(self.averaging_method):
            combined = self.averaging_method(preds)
        elif self.averaging_method == "median":
            combined = np.median(preds, axis=0)
        elif self.weights_ is not None:
            combined = np.average(preds, weights=self.weights_, axis=0)
        else:
            combined = np.mean(preds, axis=0)
        return self._validate_combined_prediction(combined, preds)

    def _validate_combined_prediction(self, combined, preds):
        """Validate that a combiner returned the expected forecast shape."""
        combined = np.asarray(combined, dtype=float)
        expected_shape = preds.shape[1:]
        if expected_shape == ():
            if combined.shape not in ((), (1,)):
                raise ValueError(
                    "averaging_method callable must return a scalar for "
                    "one-step prediction."
                )
            return float(combined.reshape(-1)[0])
        if combined.shape != expected_shape:
            raise ValueError(
                "averaging_method callable must return an array with shape "
                f"{expected_shape}, but got {combined.shape}."
            )
        return combined

    def _validate_parameters(self):
        """Validate forecaster, averaging, and weight parameters."""
        if self.iterative_strategy not in ("component", "ensemble"):
            raise ValueError(
                "iterative_strategy must be one of {'component', 'ensemble'}, "
                f"but found {self.iterative_strategy!r}."
            )

        if not callable(self.averaging_method) and self.averaging_method not in (
            "mean",
            "median",
        ):
            raise ValueError(
                "averaging_method must be 'mean', 'median', or a callable, "
                f"got {self.averaging_method!r}"
            )

        for name, forecaster in self._forecasters:
            if self.iterative_strategy == "component" and not callable(
                getattr(forecaster, "iterative_forecast", None)
            ):
                raise TypeError(
                    f"Forecaster {name!r} must implement iterative_forecast when "
                    "iterative_strategy='component'."
                )

        if self.averaging_method != "mean" or self.weights is None:
            return None

        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional array.")
        if weights.shape[0] != len(self._forecasters):
            raise ValueError(
                f"weights has length {weights.shape[0]} but there are "
                f"{len(self._forecasters)} forecasters."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("All weights must be finite.")
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative.")
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("At least one weight must be positive.")
        return weights / weight_sum

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
