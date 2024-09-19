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

Sklearn and pandas adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer

Series-to-tabular transformers
-------------------------------

Series-to-tabular transformers transform individual time series to a vector of
features, usually a vector of floats, but can also be categorical.

When applied to collections or hierarchical data, the transformation result is a table
with as many rows as time series in the collection and a column for each feature.


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
    MiniRocketMultivariateVariable
    MultiRocket

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


Series transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations apply a function element-wise.

Depending on the transformer, the transformation parameters can be fitted.

.. currentmodule:: aeon.transformations.series._boxcox

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxTransformer
    LogTransformer

Detrending
~~~~~~~~~~

.. currentmodule:: aeon.transformations.series._clear_sky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClearSky


Filtering and denoising
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.series._bkfilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BKFilter

.. currentmodule:: aeon.transformations.series._dft

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DFTSeriesTransformer

.. currentmodule:: aeon.transformations.series._sg

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SGSeriesTransformer

.. currentmodule:: aeon.transformations.series._siv

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SIVSeriesTransformer

Slope
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.transformations.collection.slope

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SlopeTransformer

Segmentation
~~~~~~~~~~~~

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
    SlidingWindowSegmenter

Collection transformers
~~~~~~~~~~~~~~~~~~~~~~~

Collection transformers transform a collection of time series into a different
collection of time series.

.. currentmodule:: aeon.transformations.collection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HOG1DTransformer
    AutocorrelationFunctionTransformer
    ARCoefficientTransformer
    DownsampleTransformer
    DWTTransformer
    MatrixProfile
    PeriodogramTransformer
    SlopeTransformer
    TimeSeriesScaler
    Normalise

Channel selection
~~~~~~~~~~~~~~~~~

Channel selection transformers select a subset of channels from a collection of
multivariate time series.

.. currentmodule:: aeon.transformations.collection.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ChannelScorer
    ElbowClassSum
    ElbowClassPairwise
    RandomChannelSelector


Equal length transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations convert a collection of unequal length series into
collections of equal length series.

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

.. currentmodule:: aeon.transformations.series._pca

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PCATransformer


Outlier detection, changepoint detection
----------------------------------------

.. currentmodule:: aeon.transformations.series._clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPTransformer
