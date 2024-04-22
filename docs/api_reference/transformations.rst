.. _transformations_ref:

Transformations
===============

The :mod:`aeon.transformations` module contains classes for data
transformations.

.. automodule:: aeon.transformations
   :no-members:
   :no-inherited-members:

All (simple) transformers in `aeon` can be listed using the `aeon.registry
.all_estimators` utility, using `estimator_types="transformer"` tag.

Transformations are categorized as follows:

.. list-table::
   :header-rows: 1

   * - Category
     - Explanation
     - Example
   * - Composition
     - Building blocks for pipelines, wrappers, adapters
     - Transformer pipeline
   * - Series-to-tabular
     - Transforms series to tabular data
     - Length and mean
   * - series-to-series
     - Transforms individual series to series
     - Differencing, detrending
   * - Series-to-collection
     - transforms a series into a collection of time series
     - Bootstrap, sliding window
   * - Collection
     - Transforms a collection of times series into a new collection of time series
     - Padding to equal length
   * - Hierarchical
     - uses hierarchy information non-trivially
     - Reconciliation

Composition
-----------

Pipeline building
~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TransformerPipeline
    FeatureUnion
    ColumnwiseTransformer
    ColumnTransformer
    FitInTransform
    MultiplexTransformer
    OptionalPassthrough
    InvertTransform
    Id
    YtoX

.. currentmodule:: aeon.transformations.func_transform

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FunctionTransformer

Sklearn and pandas adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer

.. currentmodule:: aeon.transformations.adapt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TabularToSeriesAdaptor
    PandasTransformAdaptor

Series-to-tabular transformers
-------------------------------

Series-to-tabular transformers transform individual time series to a vector of
features, usually a vector of floats, but can also be categorical.

When applied to collections or hierarchical data, the transformation result is a table
with as many rows as time series in the collection and a column for each feature.

Summarization
~~~~~~~~~~~~~

These transformers extract simple summary features.

.. currentmodule:: aeon.transformations.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SummaryTransformer
    WindowSummarizer
    FittedParamExtractor


Shapelets, wavelets and convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomShapeletTransform
    RandomDilatedShapeletTransform
    SAST

.. currentmodule:: aeon.transformations.collection.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket
    MiniRocket
    MiniRocketMultivariate
    MiniRocketMultivariateVariable
    MultiRocket
    MultiRocketMultivariate

.. currentmodule:: aeon.transformations.collection.dwt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DWTTransformer

Distance-based features
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfile

Dictionary-based features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PAA
    SFA
    SAX

Signature-based features
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.signature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureTransformer

Feature collections
~~~~~~~~~~~~~~~~~~~

These transformers extract larger collections of features.

.. currentmodule:: aeon.transformations.collection.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSFreshRelevantFeatureExtractor
    TSFreshFeatureExtractor
    Catch22

Series-to-series transformers
-----------------------------

Series-to-series transformers transform individual time series into another time series.
When applied to collections or hierarchical data, individual series are transformed
through broadcasting.

Lagging
~~~~~~~

.. currentmodule:: aeon.transformations.lag

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Lag

Element-wise transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations apply a function element-wise.

Depending on the transformer, the transformation parameters can be fitted.

.. currentmodule:: aeon.transformations.boxcox

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxTransformer
    LogTransformer

.. currentmodule:: aeon.transformations.scaledlogit

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScaledLogitTransformer

.. currentmodule:: aeon.transformations.cos

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CosineTransformer

.. currentmodule:: aeon.transformations.exponent

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentTransformer
    SqrtTransformer

Detrending
~~~~~~~~~~

.. currentmodule:: aeon.transformations.detrend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer
    STLTransformer

.. currentmodule:: aeon.transformations.clear_sky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClearSky


Filtering and denoising
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.bkfilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BKFilter

.. currentmodule:: aeon.transformations.kalman_filter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KalmanFilterTransformer

.. currentmodule:: aeon.transformations.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaLinesTransformer

Differencing and slope
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.difference

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Differencer

.. currentmodule:: aeon.transformations.collection.slope

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SlopeTransformer

Binning and segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.binning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeBinAggregate

.. currentmodule:: aeon.transformations.collection.interpolate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSInterpolator

.. currentmodule:: aeon.transformations.collection.segment

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

Missing value imputation
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.impute

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Imputer

Seasonality and Date-Time Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.date

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DateTimeFeatures

.. currentmodule:: aeon.transformations.time_since

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSince

.. currentmodule:: aeon.transformations.fourier

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FourierFeatures

Auto-correlation series
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.acf

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationTransformer
    PartialAutoCorrelationTransformer

Window-based series transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers create a series based on a sequence of sliding windows.

.. currentmodule:: aeon.transformations.collection.hog1d

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HOG1DTransformer

Multivariate-to-univariate
~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers convert multivariate series to univariate.

.. currentmodule:: aeon.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnConcatenator

Augmentation
~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.augmenter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    InvertAugmenter
    RandomSamplesAugmenter
    ReverseAugmenter
    WhiteNoiseAugmenter

FeatureSelection
~~~~~~~~~~~~~~~~

These transformers select features in `X` based on `y`.

.. currentmodule:: aeon.transformations.feature_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FeatureSelection

.. currentmodule:: aeon.transformations.collection.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ElbowClassSum
    ElbowClassPairwise

Subsetting time points and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers subset `X` by time points (`pandas` index or index level) or variables (`pandas` columns).

.. currentmodule:: aeon.transformations.subset

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnSelect
    IndexSubset

Panel transformers
------------------

Panel transformers transform a panel of time series into a panel of time series.

A panel transformer is fitted on an entire panel, and not per series.

Equal length transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations ensure all series in a panel have equal length

.. currentmodule:: aeon.transformations.collection.pad

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PaddingTransformer

.. currentmodule:: aeon.transformations.collection.truncate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TruncationTransformer


Dimension reduction
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.pca

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PCATransformer

Series-to-Panel transformers
----------------------------

These transformers create a panel from a single series.

Bootstrap transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.bootstrap

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    STLBootstrapTransformer
    MovingBlockBootstrapTransformer

Outlier detection, changepoint detection
----------------------------------------

.. currentmodule:: aeon.transformations.outlier_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HampelFilter

.. currentmodule:: aeon.transformations.clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPTransformer

Hierarchical transformers
-------------------------

These transformers are specifically for hierarchical data and panel data.

The transformation depends on the specified hierarchy in a non-trivial way.

.. currentmodule:: aeon.transformations.hierarchical.aggregate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Aggregator

.. currentmodule:: aeon.transformations.hierarchical.reconcile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Reconciler
