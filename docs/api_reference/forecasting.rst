.. _forecasting_ref:

Forecasting
===========

The :mod:`aeon.forecasting` module contains algorithms for forecasting.

All forecasters in ``aeon``  can be listed using the ``aeon.utils.discovery
.all_estimators`` function using ``type_filter="forecaster"``, optionally filtered by
tags. Valid tags for forecasters can be found with ``aeon.utils.tags
.all_tags_for_estimator`` function with the argument ``"forecaster"``.

Forecasting Models
------------------

.. currentmodule:: aeon.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseForecaster
    NaiveForecaster
    RegressionForecaster
    TVPForecaster

Statistical Models
------------------

.. currentmodule:: aeon.forecasting.stats

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ARIMA
    ETS
