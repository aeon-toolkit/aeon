.. _regression_ref:

Time series regression
======================

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

Deep learning
-------------

.. currentmodule:: aeon.regression.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CNNRegressor
    TapNetRegressor
    InceptionTimeRegressor
    IndividualInceptionRegressor

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

    FreshPRINCERegressor


Interval-based
--------------

.. currentmodule:: aeon.regression.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesForestRegressor

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
