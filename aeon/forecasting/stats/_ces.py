"""Complex Exponential Smoothing forecaster."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ComplexExponentialSmoothing"]


import numpy as np

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin


class ComplexExponentialSmoothing(BaseForecaster, IterativeForecastingMixin):
    """Complex Exponential Smoothing forecaster.

    This is an initial scaffold for non-seasonal Complex Exponential Smoothing
    (CES). The full CES state-space recursion and parameter optimisation are
    not implemented yet.

    Parameters
    ----------
    alpha_real : float or None, default=None
        Real part of the complex smoothing parameter. If ``None``, it will be
        estimated during fitting in the full implementation.
    alpha_imag : float or None, default=None
        Imaginary part of the complex smoothing parameter. If ``None``, it will
        be estimated during fitting in the full implementation.
    initial_level : float or None, default=None
        Initial level/state value. If ``None``, it will be estimated during
        fitting in the full implementation.
    alpha_real_bounds : tuple of float, default=(0.0, 1.0)
        Bounds for the real part of the smoothing parameter.
    alpha_imag_bounds : tuple of float, default=(-1.0, 1.0)
        Bounds for the imaginary part of the smoothing parameter.
    initial_level_bounds : tuple of float, default=(-1e10, 1e10)
        Bounds for the initial level.
    max_iter : int, default=500
        Maximum number of optimiser iterations.
    tol : float, default=1e-6
        Optimisation tolerance.
    """

    _tags = {
        "capability:exogenous": False,
        "python_dependencies": None,
    }

    def __init__(
        self,
        alpha_real=None,
        alpha_imag=None,
        initial_level=None,
        alpha_real_bounds=(0.0, 1.0),
        alpha_imag_bounds=(-1.0, 1.0),
        initial_level_bounds=(-1e10, 1e10),
        max_iter=500,
        tol=1e-6,
    ):
        self.alpha_real = alpha_real
        self.alpha_imag = alpha_imag
        self.initial_level = initial_level
        self.alpha_real_bounds = alpha_real_bounds
        self.alpha_imag_bounds = alpha_imag_bounds
        self.initial_level_bounds = initial_level_bounds
        self.max_iter = max_iter
        self.tol = tol

        super().__init__()

    def _fit(self, y, exog=None):
        """Fit the placeholder CES forecaster.

        Parameters
        ----------
        y : np.ndarray
            One-dimensional time series.
        exog : np.ndarray or None, default=None
            Exogenous variables. Not supported.

        Returns
        -------
        self :
            Fitted estimator.
        """
        if exog is not None:
            raise NotImplementedError(
                "ComplexExponentialSmoothing does not support exogenous " "variables."
            )

        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if y.shape[0] < 2:
            raise ValueError(
                "ComplexExponentialSmoothing requires at least two observations."
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("ComplexExponentialSmoothing requires finite values.")

        # Placeholder fitted attributes. These make the class usable as a
        # scaffold while making it clear that the real CES implementation is
        # still pending.
        self.alpha_real_ = 0.5 if self.alpha_real is None else float(self.alpha_real)
        self.alpha_imag_ = 0.0 if self.alpha_imag is None else float(self.alpha_imag)
        self.initial_level_ = (
            float(y[0]) if self.initial_level is None else float(self.initial_level)
        )

        self.complex_alpha_ = self.alpha_real_ + 1j * self.alpha_imag_

        # Naive placeholders. Replace these once the CES recursion is added.
        self.fitted_values_ = np.empty_like(y)
        self.fitted_values_[0] = self.initial_level_
        self.fitted_values_[1:] = y[:-1]

        self.residuals_ = y - self.fitted_values_
        self.sse_ = float(np.sum(self.residuals_**2))
        self.forecast_ = float(y[-1])

        return self

    def _predict(self, y, exog=None):
        """Predict one step ahead using the placeholder CES forecaster.

        Parameters
        ----------
        y : np.ndarray
            One-dimensional context series.
        exog : np.ndarray or None, default=None
            Exogenous variables. Not supported.

        Returns
        -------
        float
            One-step-ahead forecast.
        """
        if exog is not None:
            raise NotImplementedError(
                "ComplexExponentialSmoothing does not support exogenous " "variables."
            )

        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if y.shape[0] < 1:
            raise ValueError(
                "ComplexExponentialSmoothing requires at least one observation "
                "for prediction."
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("ComplexExponentialSmoothing requires finite values.")

        # Placeholder one-step forecast. Replace with CES state update.
        return float(y[-1])

    def iterative_forecast(self, y, prediction_horizon, exog=None):
        """Fit and recursively forecast multiple steps ahead.

        Parameters
        ----------
        y : np.ndarray
            One-dimensional time series.
        prediction_horizon : int
            Number of future observations to forecast.
        exog : np.ndarray or None, default=None
            Exogenous variables. Not supported.

        Returns
        -------
        np.ndarray
            Forecasts of shape ``(prediction_horizon,)``.
        """
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1.")

        self.fit(y, exog=exog)

        # Placeholder recursive forecasts. Replace with CES recursion.
        return np.full(prediction_horizon, self.forecast_, dtype=np.float64)
