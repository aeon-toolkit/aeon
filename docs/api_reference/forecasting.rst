.. _forecasting_ref:

Forecasting
===========

The :mod:`aeon.forecasting` module contains algorithms for forecasting.

All clusterers in `aeon` can be listed using the `aeon.registry.all_estimators`
utility, using `estimator_types="forecasting"`, optionally filtered by tags.
Valid tags can be listed using `aeon.registry.all_tags`.

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
