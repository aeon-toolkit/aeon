
.. _forecasting_ref:

Forecasting
===========

The :mod:`aeon.forecasting` module contains algorithms and composition tools for forecasting.

All clusterers in ``aeon``can be listed using the ``aeon.registry.all_estimators`` utility,
using ``estimator_types="forecaster"``, optionally filtered by tags.
Valid tags can be listed using ``aeon.registry.all_tags``.

Base
----

.. currentmodule:: aeon.forecasting.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseForecaster
    ForecastingHorizon

Pipeline composition
--------------------

Compositors for building forecasting pipelines.
Pipelines can also be constructed using ``*``, ``+``, and ``|`` dunders.

.. currentmodule:: aeon.pipeline

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_pipeline

.. currentmodule:: aeon.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TransformedTargetForecaster
    ForecastingPipeline
    ColumnEnsembleForecaster
    MultiplexForecaster
    ForecastX
    ForecastByLevel
    Permute
    HierarchyEnsembleForecaster

Reduction
---------

Reduction forecasters that use ``sklearn`` regressors or ``aeon`` time series regressors to make forecasts.
Use ``make_reduction`` for easy specification.

.. currentmodule:: aeon.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_reduction

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DirectTabularRegressionForecaster
    DirectTimeSeriesRegressionForecaster
    MultioutputTabularRegressionForecaster
    MultioutputTimeSeriesRegressionForecaster
    RecursiveTabularRegressionForecaster
    RecursiveTimeSeriesRegressionForecaster
    DirRecTabularRegressionForecaster
    DirRecTimeSeriesRegressionForecaster

Naive forecaster
----------------

.. currentmodule:: aeon.forecasting.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveForecaster

Prediction intervals
--------------------

Wrappers that add prediction intervals to any forecaster.

.. currentmodule:: aeon.forecasting.squaring_residuals

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SquaringResiduals

.. currentmodule:: aeon.forecasting.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveVariance

.. currentmodule:: aeon.forecasting.conformal

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ConformalIntervals

.. currentmodule:: aeon.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaggingForecaster


Trend forecasters
-----------------

.. currentmodule:: aeon.forecasting.trend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TrendForecaster
    PolynomialTrendForecaster
    STLForecaster

Exponential smoothing based forecasters
---------------------------------------

.. currentmodule:: aeon.forecasting.exp_smoothing

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentialSmoothing

.. currentmodule:: aeon.forecasting.ets

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoETS

.. currentmodule:: aeon.forecasting.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaForecaster

.. currentmodule:: aeon.forecasting.croston

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Croston

AR/MA type forecasters
----------------------

Forecasters with AR or MA component.
All "ARIMA" models below include SARIMAX capability.

.. currentmodule:: aeon.forecasting.arima

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoARIMA
    ARIMA

.. currentmodule:: aeon.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoARIMA

.. currentmodule:: aeon.forecasting.sarimax

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SARIMAX

.. currentmodule:: aeon.forecasting.var

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VAR

.. currentmodule:: aeon.forecasting.varmax

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VARMAX

Structural time series models
-----------------------------

.. currentmodule:: aeon.forecasting.bats

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BATS

.. currentmodule:: aeon.forecasting.tbats

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TBATS

.. currentmodule:: aeon.forecasting.fbprophet

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Prophet

.. currentmodule:: aeon.forecasting.structural

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UnobservedComponents

.. currentmodule:: aeon.forecasting.dynamic_factor

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DynamicFactor

Ensembles and stacking
----------------------

.. currentmodule:: aeon.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EnsembleForecaster
    AutoEnsembleForecaster
    StackingForecaster

Hierarchical reconciliation
---------------------------

.. currentmodule:: aeon.forecasting.reconcile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ReconcilerForecaster

Online and stream forecasting
-----------------------------

.. currentmodule:: aeon.forecasting.online_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    OnlineEnsembleForecaster
    NormalHedgeEnsemble
    NNLSEnsemble

.. currentmodule:: aeon.forecasting.stream

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UpdateEvery
    UpdateRefitsEvery
    DontUpdate


Model selection and tuning
--------------------------

.. currentmodule:: aeon.forecasting.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ForecastingGridSearchCV
    ForecastingRandomizedSearchCV

Model Evaluation (Backtesting)
------------------------------

.. currentmodule:: aeon.forecasting.model_evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    evaluate

Time series splitters
---------------------

Time series splitters can be used in both evaluation and tuning.

.. currentmodule:: aeon.forecasting.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CutoffSplitter
    SingleWindowSplitter
    SlidingWindowSplitter
    ExpandingWindowSplitter

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    temporal_train_test_split
