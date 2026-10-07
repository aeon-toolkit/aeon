# Getting Started

The following information is designed to get users up and running with `aeon` quickly.
If installation is required, please see our [installation guide](installation.md) for
installing `aeon`.

We assume basic familiarity with the [`scikit-learn`](https://scikit-learn.org/stable/index.html)
package. If you want help with `scikit-learn` you may want to view
[their getting started guides](https://scikit-learn.org/stable/getting_started.html).

`aeon` is an open-source toolkit for learning from time series. It provides access to
the very latest algorithms for time series machine learning, in addition to a range of
classical techniques for the following learning tasks:

- [**Classification**](api_reference/classification), where a collection of time series
  labelled with  a discrete value is used to train a model to predict unseen cases
  ([more details](examples/classification/classification.ipynb)).
- [**Regression**](api_reference/regression), where a collection of time series
  labelled with a continuous value is used to train a model to predict unseen cases
  ([more details](examples/regression/regression.ipynb)).
- [**Clustering**](api_reference/clustering), where a collection of time series without
  any labels are used to train a model to label cases
  ([more details](examples/clustering/clustering.ipynb)).
- [**Similarity search**](api_reference/similarity_search), where the goal is to find
  nearest neighbors among subsequences or whole series within a collection of time
  series in an efficient way.
  ([more details](examples/similarity_search/similarity_search.ipynb)).
- [**Anomaly detection**](api_reference/anomaly_detection), where the goal is to find
  values or areas of a single time series that are not representative of the whole series
  ([more details](examples/anomaly_detection/anomaly_detection.ipynb)).
- [**Forecasting**](api_reference/forecasting.rst), where the goal is to predict future values
  of a single time series
  ([more details](examples/forecasting/forecasting.ipynb)).
- [**Segmentation**](api_reference/segmentation), where the goal is to split a single time
  series into regions that are dissimilar to each other
  ([more details](examples/segmentation/segmentation.ipynb)).

`aeon` also provides core modules that are used by the modules above:

- [**Transformations**](api_reference/transformations), where either a single series or collection is
  transformed into a different representation or domain. ([more details](examples/transformations/transformations.ipynb)).
- [**Distances**](api_reference/distances), which measure the dissimilarity between two time series or
  collections of series and include functions to align series ([more details](examples/distances/distances.ipynb)).
- [**Networks**](api_reference/networks), provides core models for deep learning for all time series tasks
  ([more details](examples/networks/deep_learning.ipynb)).

This guide gives the briefest of introductions to the main concepts and code for each
task. There are dedicated notebooks going into more detail for each module, see the
links above, the [API](api_reference) and the [examples](examples.md) pages.

## A Single Time Series

A time series is a series of real valued data assumed to be ordered. A univariate
time series has a single value at each time point, for example the heartbeat ECG
reading from a single sensor. A multivariate time series is made up of multiple
channels, where each observation is a vector of related recordings at the same time
index, for example the X, Y and Z coordinates of a motion trace from a smartwatch.

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/series.svg
```
A univariate series stores one value per time point, shape `(n_timepoints,)`. A
multivariate series has one row per channel, shape `(n_channels, n_timepoints)`.
:::

Single time series are stored by default in a `np.ndarray`. The airline series is a
classic univariate example: the monthly totals of international airline passengers,
1949 to 1960, in thousands.

```python
from aeon.datasets import load_airline, load_uschange

y = load_airline()  # univariate series, shape (144,)
X = load_uschange()  # multivariate series, shape (5, 187)
```

We commonly refer to the number of observations for a time series as `n_timepoints`.
If a series is multivariate, we refer to the dimensions as channels (to avoid
confusion with the dimensions of array) and in code use `n_channels`. So the US Change
data loaded above has five channels and 187 time points. For more details on our
provided datasets and on how to load data into aeon compatible data structures, see
our [datasets](examples/datasets/datasets.ipynb) notebooks.

:::{dropdown} Other input types and the axis parameter
We can also handle `pd.Series` and `pd.DataFrame` objects as inputs, but these may be
converted to `np.ndarray` internally.

Single multivariate series input typically follows the shape
`(n_channels, n_timepoints)` by default. Algorithms may have an `axis` parameter to
change this, where `axis=1` assumes the default shape and is the default setting, and
`axis=0` assumes the shape `(n_timepoints, n_channels)`.
:::

## Single Series Modules

Different `aeon` modules work with individual series or collections of series.
Estimators in the `forecasting` and `segmentation` modules and the series detectors of
the `anomaly detection` module use single series input (they inherit from
`BaseSeriesEstimator`). The functions in `distances` take two series as arguments.

:::::{tab-set}

::::{tab-item} Anomaly detection
Anomaly detection (AD) is the process of identifying observations that are
significantly different from the rest of the data. The detectors for single series are
in `aeon.anomaly_detection.series`, while `aeon.anomaly_detection.collection` contains
detectors which flag whole series of a collection as anomalous. This example uses
`STOMP`, which requires the `stumpy` soft dependency.

```python
from aeon.anomaly_detection.series.distance_based import STOMP
from aeon.datasets import load_airline

y = load_airline()
stomp = STOMP(window_size=12)
scores = stomp.fit_predict(y)  # one anomaly score per time point, shape (144,)
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/anomaly_detection.svg
```
The [anomaly score]{.k-b} has one value per time point of the [series]{.k-a}. It is
highest in 1949, the shaded year, which looks the least like the other years.
:::

More details in the [anomaly detection notebook](examples/anomaly_detection/anomaly_detection.ipynb).
::::

::::{tab-item} Forecasting
Forecasting is the task of predicting future values of a time series. Forecasters
inherit from [BaseForecaster](forecasting.BaseForecaster). They use `fit` to learn a
model from a series, then `predict` to predict the value `horizon` steps after its
end, and `forecast` does both in one call. Forecasters which can feed their own
predictions back as input also have an `iterative_forecast` method. Here we use an
[ETS](forecasting.stats.ETS) exponential smoothing model.

```python
from aeon.datasets import load_airline
from aeon.forecasting.stats import ETS

y = load_airline()
ets = ETS(trend_type="additive", seasonality_type="multiplicative", seasonal_period=12)
ets.forecast(y)  # fit on y and predict the next month: 452.5
ets.iterative_forecast(y, prediction_horizon=6)  # predict the next six months
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/forecasting.svg
```
The last five years of the [airline series]{.k-a}, in thousands of passengers, and the
[six months forecast by ETS]{.k-b}.
:::

More details in the [forecasting notebook](examples/forecasting/forecasting.ipynb).
::::

::::{tab-item} Segmentation
Time series segmentation (TSS) is the process of dividing a time series into regions
that are dissimilar to each other, for example splitting the motion trace from a
smartwatch into walking, running and sitting. It is closely related to change point
detection, a term used more in the statistics literature.

```python
from aeon.datasets import load_airline
from aeon.segmentation import ClaSPSegmenter

y = load_airline()
clasp = ClaSPSegmenter()
clasp.fit_predict(y)  # the change points of the series: [51]
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/segmentation.svg
```
ClaSP finds one change point, at index 51 (April 1953). It splits the series into
[a first]{.k-a} and [a second]{.k-b} region.
:::

More details in the [segmentation notebook](examples/segmentation/segmentation.ipynb).
::::

::::{tab-item} Distances
Distances between time series is a primitive operation in very many time series
tasks. We have an extensive set of distance functions in the `aeon.distances` module,
all optimised using numba. They all work with multivariate and unequal length series.

```python
from aeon.datasets import load_japanese_vowels
from aeon.distances import dtw_distance

X, y = load_japanese_vowels()  # multivariate series of unequal length
dtw_distance(X[0], X[1])  # 14.42
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/distances.svg
```
DTW matches each time point of the [first series]{.k-a} (20 points) with one or more
points of the [second]{.k-b} (26 points). The path is computed over the 12 channels,
we show the first one.
:::

More details in the [distances notebook](examples/distances/distances.ipynb).
::::

:::::

## Collections of Time Series

The default storage for collections of time series is a 3D `np.ndarray` of shape
`(n_cases, n_channels, n_timepoints)`. If `n_timepoints` varies between cases, we store
the collection in a `list` of 2D `np.ndarray`, each with the same number of channels.

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/collection.svg
```
An equal length collection is one 3D array of shape
`(n_cases, n_channels, n_timepoints)`. An unequal length collection is a list of 2D
arrays of shape `(n_channels, n_timepoints)`.
:::

```python
from aeon.datasets import load_basic_motions, load_italy_power_demand
from aeon.datasets import load_japanese_vowels

X, y = load_italy_power_demand()  # univariate, equal length: (1096, 1, 24)
X2, y2 = load_basic_motions()  # multivariate, equal length: (80, 6, 100)
X3, y3 = load_japanese_vowels()  # multivariate, unequal length: list of 640 arrays
X3[0].shape  # (12, 20)
```

We use the terms case and instance interchangeably when referring to a single time
series contained in a collection. The size of a collection is referred to as `n_cases`
in code. We do not have the capability to use collections with varying numbers of
channels, and we assume series length is the same for all channels of a single series.

:::{warning}
We recommend storing collections in a 3D `np.ndarray` even if each time series is
univariate (i.e. `n_channels == 1`). Collection estimators will work with 2D input of
shape `(n_cases, n_timepoints)` as you would expect from `scikit-learn`, but it is
possible to confuse it with a single multivariate series of shape
`(n_channels, n_timepoints)`. This potential confusion is one reason we make the
distinction between series and collection estimators.
:::

## Collection based modules

The estimators in the `classification`, `regression` and `clustering` modules learn
from collections of time series (they inherit from `BaseCollectionEstimator`), often
with an array of target variables. The `similarity_search` module also works with
collections. Collection estimators closely follow the `scikit-learn` estimator
interface (`fit`, `predict`, `transform`, `predict_proba`, `fit_predict` and
`fit_transform` where appropriate) and work directly with `scikit-learn` model
evaluation, parameter searching and pipelines.

:::::{tab-set}

::::{tab-item} Classification
Time series classification (TSC) involves training a model on a labelled collection
of time series. The labels, referred to as `y` in code, should be a `numpy` array of
type `int` or `str`. Here we fit a
[RocketClassifier](classification.convolution_based.RocketClassifier), a fast and
accurate place to start.

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_italy_power_demand

X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")

clf = RocketClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  # accuracy on new data, about 0.97
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/classification.svg
```
Each series is one day of electricity demand, from [October to March]{.k-a} or from
[April to September]{.k-b}. Bold lines are the class averages: winter days have
their evening peak earlier.
:::

Like `scikit-learn`, `predict` gives the labels of new cases and `predict_proba` the
class probabilities. More details in the
[classification notebook](examples/classification/classification.ipynb).
::::

::::{tab-item} Regression
Time series regression assumes that the target variable is continuous, the `y` array
should be of type `float`. The term is also used in forecasting with a sliding window.
Here it means "time series extrinsic regression", where the target is not future
values but some external variable. We use a
[RocketRegressor](regression.convolution_based.RocketRegressor) on the
[Covid3Month](https://zenodo.org/record/3902690) problem.

```python
from aeon.datasets import load_covid_3month
from aeon.regression.convolution_based import RocketRegressor

X_train, y_train = load_covid_3month(split="train")
X_test, y_test = load_covid_3month(split="test")

reg = RocketRegressor(random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)  # one value per series
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/regression.svg
```
The first three test series. Each one is the daily number of confirmed COVID-19 cases
in a country over three months, the value to predict is the death rate.
:::

More details in the [regression notebook](examples/regression/regression.ipynb).
::::

::::{tab-item} Clustering
Like classification and regression, time series clustering (TSCL) aims to follow the
`scikit-learn` interface where possible, with the same input data format. This example
fits a [TimeSeriesKMeans](clustering.TimeSeriesKMeans) clusterer on the
[ArrowHead](http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead)
dataset.

```python
from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_arrow_head

X, y = load_arrow_head()
kmeans = TimeSeriesKMeans(
    n_clusters=3, distance="dtw", averaging_method="mean", random_state=0
)
kmeans.fit(X)
kmeans.labels_[:10]  # the cluster of the first ten series
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/clustering.svg
```
The three clusters found in the 211 series, without using their labels. Thin lines
are series of the cluster, the bold line is its centre.
:::

After calling `fit`, the `labels_` attribute contains the cluster of each series and
`predict` gives the cluster of new data. More details in the
[clustering notebook](examples/clustering/clustering.ipynb).
::::

::::{tab-item} Similarity search
The similarity search estimators find the nearest neighbors of a query, either among
subsequences ([BaseSubsequenceSearch](similarity_search.subsequence.BaseSubsequenceSearch),
for example `MASS`) or among whole series
([BaseWholeSeriesSearch](similarity_search.whole_series.BaseWholeSeriesSearch), for
example `NaiveSeriesSearch` and the approximate `SimHashIndexANN`). They can be used
standalone or as parts of pipelines.

```python
import numpy as np
from aeon.similarity_search.subsequence import MASS

X = np.array([[[1, 1, 2, 4, 6, 6, 7, 5, 3, 2]]])  # collection to search in
q = np.array([[0, 1, 2, 2]])  # query subsequence of length 4
snn = MASS(length=4).fit(X)
indexes, distances = snn.predict(q, k=1)  # best match: case 0, time point 0
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/similarity_search.svg
```
The [query]{.k-b} is the closest to the
[subsequence starting at time point 0]{.k-a} of the [series]{.k-ink}.
:::

- The collection passed to `fit` has shape `(n_cases, n_channels, n_timepoints)`, the
  query passed to `predict` is a single series of shape `(n_channels, length)`.
- `predict` gives the `(case, timestamp)` indexes of the best matching subsequences
  and their distances to the query.

:::{dropdown} Searching for whole series
With an estimator from `aeon.similarity_search.whole_series`, the query is a complete
series and `predict` returns the indexes of the nearest whole series in the collection.

```python
import numpy as np
from aeon.similarity_search.whole_series import NaiveSeriesSearch

X = np.array([[[1, 2, 3, 4, 5]], [[1, 1, 2, 4, 6]], [[5, 4, 3, 2, 1]]])
q = np.array([[1, 2, 3, 4, 5]])  # query series of the same length
wnn = NaiveSeriesSearch().fit(X)
indexes, distances = wnn.predict(q, k=1)  # the closest whole series
```
:::

More details in the [similarity search notebook](examples/similarity_search/similarity_search.ipynb).
::::

:::::

## Transformers

We split transformers into two categories: those that transform single time series
and those that transform a collection.

### Transformers for Single Time Series

Transformers inheriting from the [BaseSeriesTransformer](transformations.series.base.BaseSeriesTransformer)
in the `aeon.transformations.series` package transform a single (possibly multivariate)
time series into a different time series or a feature vector. Here the
[AutoCorrelationSeriesTransformer](transformations.series.AutoCorrelationSeriesTransformer)
extracts the autocorrelation terms of a series.

```python
from aeon.datasets import load_airline
from aeon.transformations.series import AutoCorrelationSeriesTransformer

y = load_airline()
acf = AutoCorrelationSeriesTransformer()
res = acf.fit_transform(y)  # one value per lag, shape (1, 36)
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/acf.svg
```
Autocorrelation of the airline series for lags 1 to 36. The
[peaks every 12 months]{.k-b} come from the yearly seasonality.
:::

The [smoothing filters notebook](examples/transformations/smoothing_filters.ipynb)
shows more examples of series transformers.

### Transformers for Collections of Time Series

The `aeon.transformations.collection` module contains a range of transformers for
collections of time series. These do not allow for single series input, treat 2D input
types as a collection of univariate series, and have no restrictions on the datatype
of output.

Most time series classification and regression algorithms are based on some form of
transformation into an alternative feature space. For example,
[Catch22](transformations.collection.feature_based.Catch22) calculates 22 summary
statistics for each series, on which we can fit a traditional classifier or regressor.

```python
import numpy as np
from aeon.transformations.collection.feature_based import Catch22

X = np.random.RandomState(0).random(size=(4, 1, 10))  # four cases of 10 time points
c22 = Catch22(replace_nans=True)
c22.fit_transform(X).shape  # four cases of 22 features: (4, 22)
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/catch22.svg
```
Catch22 turns each series into 22 features, so four series become an array of shape
`(4, 22)`. The stronger the colour of a cell, the higher the value of the feature.
:::

There are also series-to-series transformations, such as the
[Padder](transformations.collection.unequal_length.Padder) to lengthen series and
process unequal length collections.

```python
from aeon.testing.data_generation import make_example_3d_numpy_list
from aeon.transformations.collection.unequal_length import Padder

X, _ = make_example_3d_numpy_list(  # two series of 12 and 9 time points
    n_cases=2, min_n_timepoints=8, max_n_timepoints=12, random_state=0
)
pad = Padder(padded_length=12, fill_value=0)
pad.fit_transform(X).shape  # one 3D array: (2, 1, 12)
```

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/padder.svg
```
The second series has 9 time points. The `Padder` adds [three zeros]{.k-b} so that
both series have 12.
:::

## Pipelines for aeon estimators

Like `scikit-learn`, `aeon` provides pipeline classes which can be used to chain
transformations and estimators together. For machine learning tasks such as
classification, regression and clustering, the `scikit-learn` `make_pipeline`
functionality can be used if the transformer outputs a valid input type.

:::{div} aeon-fig
```{raw} html
:file: images/getting_started/pipeline.svg
```
The transformer turns each series into features, then the classifier predicts a label
from them.
:::

The following example uses the [Catch22](transformations.collection.feature_based.Catch22)
feature extraction transformer and a random forest classifier to classify.

```python
from aeon.datasets import load_italy_power_demand
from aeon.transformations.collection.feature_based import Catch22
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")

pipe = make_pipeline(
    Catch22(replace_nans=True),
    RandomForestClassifier(random_state=42),
)
pipe.fit(X_train, y_train)
# make predictions like any other sklearn estimator
accuracy_score(pipe.predict(X_test), y_test)  # 0.88
```

:::{dropdown} Tuning parameters with GridSearchCV
Like with pipelines, tasks such as classification, regression and clustering can use
the available `scikit-learn` functionality.

```python
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_italy_power_demand
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold

X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")

knn = KNeighborsTimeSeriesClassifier()
param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}

gscv = GridSearchCV(knn, param_grid, cv=KFold(n_splits=4))
gscv.fit(X_train, y_train)

gscv.best_params_  # {'distance': 'euclidean', 'n_neighbors': 5}
accuracy_score(y_test, gscv.predict(X_test))  # 0.95
```
:::

## Next steps

If you do not know which estimator to pick for your task, these are good first
choices. The [estimator overview](estimator_overview.md) lists all of them.

| Task | A good first estimator | Import from |
|---|---|---|
| Classification | [RocketClassifier](classification.convolution_based.RocketClassifier) | `aeon.classification.convolution_based` |
| Regression | [RocketRegressor](regression.convolution_based.RocketRegressor) | `aeon.regression.convolution_based` |
| Clustering | [TimeSeriesKMeans](clustering.TimeSeriesKMeans) | `aeon.clustering` |
| Anomaly detection | [STOMP](anomaly_detection.series.distance_based.STOMP) | `aeon.anomaly_detection.series.distance_based` |
| Forecasting | [ETS](forecasting.stats.ETS) | `aeon.forecasting.stats` |
| Segmentation | [ClaSPSegmenter](segmentation.ClaSPSegmenter) | `aeon.segmentation` |
| Similarity search | [MASS](similarity_search.subsequence.MASS) | `aeon.similarity_search.subsequence` |

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Examples
:link: examples
:link-type: doc
:shadow: none

One notebook per module, from the basics to advanced use.
:::

:::{grid-item-card} API reference
:link: api_reference
:link-type: doc
:shadow: none

Every estimator and function, with its parameters.
:::

:::{grid-item-card} Estimator overview
:link: estimator_overview
:link-type: doc
:shadow: none

Search all estimators by module and by capability.
:::

:::{grid-item-card} Loading data
:link: examples/datasets/datasets
:link-type: doc
:shadow: none

The data structures and how to load your own datasets.
:::

:::{grid-item-card} Benchmarking
:link: examples/benchmarking/benchmarking
:link-type: doc
:shadow: none

Compare algorithms and reproduce published results.
:::

:::{grid-item-card} Get help
:link: https://discord.gg/D6rzqHGKRJ
:shadow: none

Ask questions and talk to the developers on Discord.
:::

::::
