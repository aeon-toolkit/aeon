
.. _performance_metric_ref:

Performance metrics
===================

The :mod:`aeon.performance_metrics` module contains metrics for evaluating and tuning time series models.

.. automodule:: aeon.performance_metrics
    :no-members:
    :no-inherited-members:

Forecasting
-----------

.. currentmodule:: aeon.performance_metrics.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_forecasting_scorer
    mean_absolute_scaled_error
    median_absolute_scaled_error
    mean_squared_scaled_error
    median_squared_scaled_error
    mean_absolute_error
    mean_squared_error
    median_absolute_error
    median_squared_error
    geometric_mean_absolute_error
    geometric_mean_squared_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    mean_relative_absolute_error
    median_relative_absolute_error
    geometric_mean_relative_absolute_error
    geometric_mean_relative_squared_error
    mean_asymmetric_error
    mean_linex_error
    relative_loss


Segmentation
~~~~~~~~~~~~

.. currentmodule:: aeon.performance_metrics.segmentation.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    count_error
    hausdorff_error
    prediction_ratio
