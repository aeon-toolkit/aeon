.. _regression_ref:

Regression
==========

The :mod:`aeon.regression` module contains algorithms and composition tools for time series regression.

All regressors in ``aeon``can be listed using the ``aeon.registry.all_estimators`` utility,
using ``estimator_types="regressor"``, optionally filtered by tags.
Valid tags can be listed using ``aeon.registry.all_tags``.

Convolution-based
-----------------

.. currentmodule:: aeon.regression.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RocketRegressor
    HydraRegressor
    MultiRocketHydraRegressor

Deep learning
-------------

.. currentmodule:: aeon.regression.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CNNRegressor
    FCNRegressor
    TapNetRegressor
    InceptionTimeRegressor
    IndividualInceptionRegressor
    ResNetRegressor

Distance-based
--------------

.. currentmodule:: aeon.regression.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KNeighborsTimeSeriesRegressor

Dummy
-----

.. currentmodule:: aeon.regression

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyRegressor

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

    RISTRegression


Interval-based
--------------

.. currentmodule:: aeon.regression.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesForestRegressor

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
