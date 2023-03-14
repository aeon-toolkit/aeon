.. _classification_ref:

Time series classification
==========================

The :mod:`aeon.classification` module contains algorithms and composition tools for time series classification.

All classifiers in ``aeon`` can be listed using the ``aeon.registry
.all_estimators`` utility,
using ``estimator_types="classifier"``, optionally filtered by tags.
Valid tags can be listed using ``aeon.registry.all_tags``.

Composition
-----------

.. currentmodule:: aeon.classification.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClassifierPipeline
    ChannelEnsembleClassifier
    ComposableTimeSeriesForestClassifier
    SklearnClassifierPipeline
    WeightedEnsembleClassifier

Deep learning
-------------

.. currentmodule:: aeon.classification.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CNNClassifier
    FCNClassifier
    MLPClassifier
    TapNetClassifier

Dictionary-based
----------------

.. currentmodule:: aeon.classification.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BOSSEnsemble
    ContractableBOSS
    IndividualBOSS
    IndividualTDE
    MUSE
    TemporalDictionaryEnsemble
    WEASEL
    WEASEL_V2

Distance-based
--------------

.. currentmodule:: aeon.classification.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ElasticEnsemble
    KNeighborsTimeSeriesClassifier
    ShapeDTW

Early classification
--------------------

.. currentmodule:: aeon.classification.early_classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ProbabilityThresholdEarlyClassifier
    TEASER

Feature-based
-------------

.. currentmodule:: aeon.classification.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Classifier
    FreshPRINCE
    MatrixProfileClassifier
    RandomIntervalClassifier
    SignatureClassifier
    SummaryClassifier
    TSFreshClassifier

Hybrid
------

.. currentmodule:: aeon.classification.hybrid

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HIVECOTEV1
    HIVECOTEV2

Interval-based
--------------

.. currentmodule:: aeon.classification.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CanonicalIntervalForest
    DrCIF
    RandomIntervalSpectralEnsemble
    SupervisedTimeSeriesForest
    TimeSeriesForestClassifier

Kernel-based
------------

.. currentmodule:: aeon.classification.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Arsenal
    RocketClassifier

Shapelet-based
--------------

.. currentmodule:: aeon.classification.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransformClassifier

sklearn
-------

.. currentmodule:: aeon.classification.sklearn

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ContinuousIntervalTree
    RotationForest

Base
----

.. currentmodule:: aeon.classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClassifier
    DummyClassifier

.. currentmodule:: aeon.classification.deep_learning.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepClassifier

.. currentmodule:: aeon.classification.early_classification.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseEarlyClassifier
