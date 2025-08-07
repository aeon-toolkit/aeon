.. _classification_ref:

Classification
==============

The :mod:`aeon.classification` module contains algorithms for time series classification.

All classifiers in ``aeon``  can be listed using the ``aeon.utils.discovery
.all_estimators`` function using ``type_filter="classifier"``, optionally filtered by
tags. Valid tags for classifiers can be found with ``aeon.utils.tags.all_tags_for_estimator``
function with the argument ``"classifier"``.


Convolution-based
-----------------

.. currentmodule:: aeon.classification.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Arsenal
    HydraClassifier
    MultiRocketHydraClassifier
    RocketClassifier
    MiniRocketClassifier
    MultiRocketClassifier

Deep learning
-------------

.. currentmodule:: aeon.classification.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeCNNClassifier
    EncoderClassifier
    FCNClassifier
    InceptionTimeClassifier
    IndividualInceptionClassifier
    IndividualLITEClassifier
    LITETimeClassifier
    MLPClassifier
    ResNetClassifier
    DisjointCNNClassifier

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
    MrSEQLClassifier
    MrSQMClassifier
    MUSE
    REDCOMETS
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
    ProximityForest
    ProximityTree

Feature-based
-------------

.. currentmodule:: aeon.classification.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Classifier
    FreshPRINCEClassifier
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
    RISTClassifier

Interval-based
--------------

.. currentmodule:: aeon.classification.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CanonicalIntervalForestClassifier
    DrCIFClassifier
    IntervalForestClassifier
    QUANTClassifier
    RandomIntervalClassifier
    RandomIntervalSpectralEnsembleClassifier
    RSTSF
    SupervisedIntervalClassifier
    SupervisedTimeSeriesForest
    TimeSeriesForestClassifier

Shapelet-based
--------------

.. currentmodule:: aeon.classification.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    LearningShapeletClassifier
    RDSTClassifier
    SASTClassifier
    RSASTClassifier
    ShapeletTransformClassifier

sklearn
-------

.. currentmodule:: aeon.classification.sklearn

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ContinuousIntervalTree
    RotationForestClassifier
    SklearnClassifierWrapper

Early classification
--------------------

.. currentmodule:: aeon.classification.early_classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ProbabilityThresholdEarlyClassifier
    TEASER


Ordinal classification
----------------------

.. currentmodule:: aeon.classification.ordinal_classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IndividualOrdinalTDE
    OrdinalTDE

Composition
-----------

.. currentmodule:: aeon.classification.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClassifierChannelEnsemble
    ClassifierEnsemble
    ClassifierPipeline

Dummy
-----

.. currentmodule:: aeon.classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyClassifier

Base
----

.. currentmodule:: aeon.classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClassifier

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
