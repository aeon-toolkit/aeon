.. _regression_ref:

Regression
==========

The :mod:`aeon.regression` module contains algorithms for time series regression.

All regressors in ``aeon``  can be listed using the `aeon.utils.discovery
.all_estimators` function using ``type_filter="regressor"``, optionally filtered by
tags. Valid tags for regressors can be found with ``aeon.utils.tags.all_tags_for_estimator`` function with the argument ``"regressor"``.



Convolution-based
-----------------

.. currentmodule:: aeon.regression.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HydraRegressor
    MultiRocketHydraRegressor
    RocketRegressor
    MiniRocketRegressor
    MultiRocketRegressor

Deep learning
-------------

.. currentmodule:: aeon.regression.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeCNNRegressor
    EncoderRegressor
    FCNRegressor
    InceptionTimeRegressor
    IndividualLITERegressor
    IndividualInceptionRegressor
    LITETimeRegressor
    ResNetRegressor
    MLPRegressor
    DisjointCNNRegressor
    RecurrentRegressor

Distance-based
--------------

.. currentmodule:: aeon.regression.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KNeighborsTimeSeriesRegressor

Feature-based
--------------

.. currentmodule:: aeon.regression.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Regressor
    FreshPRINCERegressor
    SummaryRegressor
    TSFreshRegressor

Hybrid
------

.. currentmodule:: aeon.regression.hybrid

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RISTRegressor


Interval-based
--------------

.. currentmodule:: aeon.regression.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CanonicalIntervalForestRegressor
    DrCIFRegressor
    IntervalForestRegressor
    RandomIntervalRegressor
    RandomIntervalSpectralEnsembleRegressor
    TimeSeriesForestRegressor
    QUANTRegressor

Shapelet-based
--------------

.. currentmodule:: aeon.regression.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RDSTRegressor


sklearn
-------

.. currentmodule:: aeon.regression.sklearn

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RotationForestRegressor
    SklearnRegressorWrapper

Compose
-------

.. currentmodule:: aeon.regression.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RegressorEnsemble
    RegressorPipeline

Dummy
-----

.. currentmodule:: aeon.regression

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyRegressor

Base
----

.. currentmodule:: aeon.regression.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseRegressor

.. currentmodule:: aeon.regression.deep_learning.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepRegressor
