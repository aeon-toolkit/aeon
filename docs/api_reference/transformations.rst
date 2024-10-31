.. _transformations_ref:

Transformations
===============

The :mod:`aeon.transformations` module contains classes for series
transformations. The module is organised into CollectionTransformers which transform a
collection of time series into a different representation and SeriesTransformers which
transform single time series.

All transformers in `aeon` can be listed using the `aeon.registry
.all_estimators` utility, using `estimator_types="transformer"` tag.


Collection transformers
-----------------------

.. currentmodule:: aeon.transformations.collection.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseCollectionTransformer

.. currentmodule:: aeon.transformations.collection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutocorrelationFunctionTransformer
    ARCoefficientTransformer
    DownsampleTransformer
    DWTTransformer
    HOG1DTransformer
    MatrixProfile
    Normalise
    Padder
    PeriodogramTransformer
    Tabularizer
    Resizer
    TimeSeriesScaler
    SlopeTransformer
    Truncator


Channel selection
~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ChannelScorer
    ElbowClassPairwise
    ElbowClassSum
    RandomChannelSelector


Compose
~~~~~~~

.. currentmodule:: aeon.transformations.collection.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CollectionTransformerPipeline


Convolution based
~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket
    MiniRocket
    MiniRocketMultivariateVariable
    MultiRocket
    HydraTransformer


Dictionary-based features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SAX
    PAA
    SFA
    SFAFast
    BORF


Feature based
~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22
    TSFresh
    TSFreshRelevant
    SevenNumberSummary


Interval based
~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomIntervals
    SupervisedIntervals
    QUANTTransformer

Shapelet based
~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomShapeletTransform
    RandomDilatedShapeletTransform
    SAST
    RSAST



Signature based
~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.signature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureTransformer


Series transforms
-----------------

.. currentmodule:: aeon.transformations.series.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSeriesTransformer


.. currentmodule:: aeon.transformations.series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationSeriesTransformer
    ClearSkyTransformer
    ClaSPTransformer
    DFTSeriesTransformer
    Dobin
    GaussSeriesTransformer
    MatrixProfileSeriesTransformer
    PLASeriesTransformer
    SGSeriesTransformer
    StatsModelsACF
    StatsModelsPACF
    BKFilter
    BoxCoxTransformer
    YeoJohnsonTransformer
    Dobin
    ScaledLogitSeriesTransformer
    SIVSeriesTransformer
    PCASeriesTransformer
    WarpingSeriesTransformer
