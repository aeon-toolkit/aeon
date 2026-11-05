.. _forecasting_ref:

Forecasting
===========

The :mod:`aeon.forecasting` module contains algorithms for forecasting.

All forecasters in ``aeon``  can be listed using the ``aeon.utils.discovery
.all_estimators`` function using ``type_filter="forecaster"``, optionally filtered by
tags. Valid tags for forecasters can be found with ``aeon.utils.tags
.all_tags_for_estimator`` function with the argument ``"forecaster"``.

Forecast output shapes
----------------------

Forecasting methods return NumPy arrays with time on the final axis. Univariate
forecasts are one-dimensional. Multivariate forecasts add a leading channel axis,
in the same channel order as the input. This convention is independent of the
input time axis and of the estimator's internal data orientation.

.. list-table:: Forecast shapes
    :header-rows: 1

    * - Method and forecast type
      - Univariate output
      - Multivariate output
    * - ``predict`` / ``forecast``: a single target time
      - ``(1,)``
      - ``(n_channels, 1)``
    * - ``predict`` / ``forecast``: a series of future values
      - ``(horizon,)``
      - ``(n_channels, horizon)``
    * - ``direct_forecast`` / ``iterative_forecast`` / ``series_to_series_forecast``
      - ``(prediction_horizon,)``
      - ``(n_channels, prediction_horizon)``

An axis is retained when the forecast length is one. A single-channel series,
including two-dimensional input with one channel, produces univariate output.
Each forecasting strategy is available only on estimators implementing its mixin;
multivariate input also requires the estimator's multivariate capability.

``horizon`` has two meanings depending on the model. A point forecaster configured
with ``horizon=3`` predicts one value three steps beyond the input history. A
series-to-series forecaster with ``horizon=3`` predicts the next three consecutive
values. ``prediction_horizon`` specifies the number of consecutive values returned
by a multi-step forecasting strategy. Iterative forecasting requires a point
forecaster with ``horizon=1``; its predictions are appended to each channel's
history along the time axis.

For example, univariate point forecasts now retain a time axis::

    >>> import numpy as np
    >>> from aeon.forecasting import NaiveForecaster
    >>> y = np.array([1., 2., 3.])
    >>> model = NaiveForecaster(strategy="drift", horizon=3)
    >>> model.forecast(y)
    array([6.])
    >>> model = NaiveForecaster(strategy="drift")
    >>> model.iterative_forecast(y, prediction_horizon=3)
    array([4., 5., 6.])

Migration from scalar predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``predict`` and ``forecast`` previously returned scalars for univariate point
forecasters. They now return arrays of shape ``(1,)``. Code that needs a Python
scalar can use ``prediction.item()``; indexing ``prediction[0]`` selects its single
value. Series forecasts retain their time axis even for ``horizon=1``. Avoid
unrestricted ``squeeze()`` when handling forecasts, since it removes singleton
time axes.


Forecasting Models
------------------

.. currentmodule:: aeon.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseForecaster
    NaiveForecaster
    RegressionForecaster

Statistical Models
------------------

.. currentmodule:: aeon.forecasting.stats

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ARIMA
    AutoARIMA
    SARIMA
    AutoSARIMA
    CES
    AutoCES
    ETS
    AutoETS
    TAR
    AutoTAR
    Theta
    DOTM
    SCUM
    TVP

Ensemble Models
---------------

.. currentmodule:: aeon.forecasting.ensembles

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EnsembleForecaster

Deep Learning Models
--------------------

.. currentmodule:: aeon.forecasting.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepForecaster
    DeepARForecaster
    TCNForecaster
    NBeatsForecaster

Machine Learning Models
------------------------

.. currentmodule:: aeon.forecasting.machine_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SETAR
    SETARForest
    SETARTree
