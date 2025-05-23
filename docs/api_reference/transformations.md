# Transformations

The `aeon.transformations` module contains classes for series
transformations. The module is organised into CollectionTransformers which transform a
collection of time series into a different representation and SeriesTransformers which
transform single time series.

All transformers in `aeon` can be listed using the `aeon.registry
.all_estimators` utility, using `estimator_types="transformer"` tag.

## Collection transformers

```{eval-rst}
.. currentmodule:: aeon.transformations.collection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutocorrelationFunctionTransformer
    ARCoefficientTransformer
    Centerer
    DownsampleTransformer
    DWTTransformer
    HOG1DTransformer
    MatrixProfile
    MinMaxScaler
    Normalizer
    Padder
    PeriodogramTransformer
    Resizer
    SlopeTransformer
    SimpleImputer
    Truncator
    Tabularizer
```

### Channel selection

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ChannelScorer
    ElbowClassPairwise
    ElbowClassSum
    RandomChannelSelector
```

### Compose

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CollectionTransformerPipeline
    CollectionId
```

### Convolution based

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.convolution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket
    MiniRocket
    MultiRocket
    HydraTransformer
```

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.convolution_based.rocketGPU

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ROCKETGPU
```

### Dictionary-based features

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SAX
    PAA
    SFA
    SFAFast
    SFAWhole
    BORF
```

### Feature based

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22
    TSFresh
    TSFreshRelevant
    SevenNumberSummary
```

### Interval based

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomIntervals
    SupervisedIntervals
    QUANTTransformer
```

### Shapelet based

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomShapeletTransform
    RandomDilatedShapeletTransform
    SAST
    RSAST
```

### Signature based

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.signature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureTransformer
```

## Series transforms

```{eval-rst}
.. currentmodule:: aeon.transformations.series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationSeriesTransformer
    ClaSPTransformer
    Dobin
    MatrixProfileSeriesTransformer
    LogTransformer
    PLASeriesTransformer
    StatsModelsACF
    StatsModelsPACF
    BKFilter
    BoxCoxTransformer
    ScaledLogitSeriesTransformer
    PCASeriesTransformer
    WarpingSeriesTransformer
```

### Compose

```{eval-rst}
.. currentmodule:: aeon.transformations.series.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SeriesTransformerPipeline
    SeriesId
```

### Smoothing

```{eval-rst}
.. currentmodule:: aeon.transformations.series.smoothing

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DiscreteFourierApproximation
    ExponentialSmoothing
    GaussianFilter
    MovingAverage
    SavitzkyGolayFilter
    RecursiveMedianSieve
```

## Base

```{eval-rst}
.. currentmodule:: aeon.transformations.collection.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseCollectionTransformer


.. currentmodule:: aeon.transformations.series.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSeriesTransformer

.. currentmodule:: aeon.transformations.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseTransformer
```
