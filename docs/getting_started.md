# Getting Started

The following information is designed to get users up and running with `aeon` quickly.
If installation is required, please see our [installation guide](installation) for
installing `aeon`.

We assume basic familiarity with the [scikit-learn](https://scikit-learn.org/stable/index.html)
package. If you want help with scikit-learn you may want to view
[their getting started guides](https://scikit-learn.org/stable/getting_started.html).

`aeon` is an open-source toolkit for learning from time series. It provides access to
the very latest algorithms for time series machine learning, in addition to a range of
classical techniques for the following learning tasks:

- {term}`Forecasting` where the goal is to predict future values of a target time
series.
- {term}`Time series classification` where the time series data for a given instance
are used to predict a categorical target class.
- {term}`Time series extrinsic regression` where the time series data for a given
instance are used to predict a continuous target value.
- {term}`Time series clustering` where the goal is to discover groups consisting of
instances with similar time series.
- {term}`Time series annotation` which is focused on outlier detection, anomaly
detection, change point detection and segmentation.
- {term}`Time series similarity search` where the goal is to evaluate the similarity
between a time series against a collection of other time series.

Additionally, it provides numerous algorithms for {term}`time series transformation`,
altering time series into different representations and domains or processing
time series data into tabular data.

The following provides introductory examples for each of these modules. The examples
use the datatypes most commonly used for the task in question, but a variety of input
types for
data are available. See [here](/examples/datasets/data_structures.ipynb) for
more information on input data structures. For more information on the variety of
estimators
available for each task, see the [API](api_reference) and [examples](examples) pages.

## Time Series Data

A time series is a series of real valued data assumed to be ordered. A univariate
time series is a singular series, where each observation is a single value. For example,
the heartbeat ECG reading from a single sensor or the number of passengers using an
airline per month would form a univariate series.

```{code-block} python
>>> from aeon.datasets import load_airline
>>> y = load_airline()  # load an example univariate series with timestamps
>>> y.head()
Period
1960-08    606.0
1960-09    508.0
1960-10    461.0
1960-11    390.0
1960-12    432.0
Freq: M, Name: Number of airline passengers, dtype: float64
```

A multivariate time series is made up of multiple series, where each observation is a
vector of related recordings in the same time index. An examples would be a motion trace
of from a smartwatch with at least three dimensions (X,Y,Z co-ordinates), or multiple
financial statistics recorded over time. Single multivariate series input typically
follows the shape `(n_timepoints, n_channels)`.

```{code-block} python
>>> from aeon.datasets import load_uschange
>>> y, X = load_uschange("Quarter")  # load an example multivariate series
>>> X.set_index(y).head()
         Consumption    Income  Production   Savings  Unemployment
Quarter
1970 Q1     0.615986  0.972261   -2.452700  4.810312           0.9
1970 Q2     0.460376  1.169085   -0.551525  7.287992           0.5
1970 Q3     0.876791  1.553271   -0.358708  7.289013           0.5
1970 Q4    -0.274245 -0.255272   -2.185455  0.985230           0.7
1971 Q1     1.897371  1.987154    1.909734  3.657771          -0.1
```

We commonly refer to the number of observations for a time series as `n_timepoints` or
`n_timepoints`. If a series is multivariate, we refer to the dimensions as channels
(to avoid confusion with the dimensions of array) and in code use `n_channels`.
Dimensions may also be referred to as variables.

Different parts of `aeon` work with single series or collections of series. The
`forecasting` and `annotation` modules will commonly use single series input, while
`classification`, `regression` and `clustering` modules will use collections of time
series. Collections of time series may also be referred to a Panels. Collections of
time series will often be accompanied by an array of target variables.

```{code-block} python
>>> from aeon.datasets import load_italy_power_demand
>>> X, y = load_italy_power_demand()  # load an example univariate collection
>>> X.shape
(1096, 1, 24)
>>> X[:5, :, :5]
[[[-0.71051757 -1.1833204  -1.3724416  -1.5930829  -1.4670021 ]]
 [[-0.99300935 -1.4267865  -1.5798843  -1.6054006  -1.6309169 ]]
 [[ 1.3190669   0.56977448  0.19512825 -0.08585642 -0.17951799]]
 [[-0.81244429 -1.1575534  -1.4163852  -1.5314215  -1.5026624 ]]
 [[-0.97284033 -1.3905178  -1.5367049  -1.6202404  -1.6202404 ]]]
>>> y[:5]
['1' '1' '2' '2' '1']
```

We use the terms case when referring to a single time series
contained in a collection. The size of a collection of time series is referred to as
`n_cases`. Collections of time typically follows the shape `
(n_cases, n_channels, n_timepoints)` if the series are equal length, but `n_timepoints`
may vary between cases.

The datatypes used by modules also differ to match the use case. Module focusing
on single series use cases will commonly use `pandas` `DataFrame` and `Series` objects
to store time series data as shown in the first two examples. Modules focusing on
collections on time series will commonly use `numpy` arrays or lists of arrays to
store time series data.

```{code-block} python
>>>from aeon.datasets import load_basic_motions, load_plaid, load_japanese_vowels
>>> X2, y2 = load_basic_motions() # example equal length multivariate collection
>>> X2.shape
(80, 6, 100)
>>> X3, y3 = load_plaid()  # example unequal length univariate collection
>>> type(X3)
<class 'list'>
>>> len(X3)
1074
>>> X3[0].shape
(1, 500)
>>> X4, y4 = load_japanese_vowels()  # example unequal length mutlivariate collection
>>> len(X4)
640
>>> X4[0].shape
(12, 20)
```

## Forecasting

The possible use cases for forecasting are more complex than with the other modules.
Like `scikit-learn`, forecasters use a fit and predict model, but the arguments are
different. The simplest forecasting use case is when you have a single series and you
want to build a model on that series (e.g. ARMA model) to predict values in the
future. At their most basic, forecasters require a series to forecast for fit, and a
forecast horizon (`fh`) to specify how many time steps ahead to make a forecast in
predict. This code fits a [TrendForecaster](forecasting.trend.TrendForecaster) on our
loaded data and predicts the next value in the series.

```{code-block} python
>>> from aeon.datasets import load_airline
>>> from aeon.forecasting.trend import TrendForecaster
>>> y = load_airline()
>>> forecaster = TrendForecaster()
>>> forecaster.fit(y)  # fit the forecaster
TrendForecaster()
>>> pred = forecaster.predict(fh=1)  # predict the next value
1961-01    472.944444
Freq: M, dtype: float64
```

An integer `fh` value will forecast *n* points into the future, i.e. a value of 3
will make a prediction for 1961-03. You can predict multiple points into the future by
passing a list of integers to `fh`.

```{code-block} python
>>> pred = forecaster.predict(fh=[1,2,3])  # predict the next 3 values
1961-01    472.944444
1961-02    475.601628
1961-03    478.258812
Freq: M, dtype: float64
```

You can split a series into train and test partitions. This code splits airline into
two parts, builds the forecasting model on the train portion and makes forecasts for
the time points represented in the test segment. In this example we use a
[ForecastingHorizon](forecasting.base.ForecastingHorizon) object to input our desired
forecast horizon using the Series index.

```{code-block} python
>>> from aeon.forecasting.model_selection import temporal_train_test_split
>>> from aeon.forecasting.base import ForecastingHorizon
>>> from aeon.performance_metrics.forecasting import mean_absolute_percentage_error
>>> y_train, y_test = temporal_train_test_split(y)  # split the data into train and test series
>>> fh = ForecastingHorizon(y_test.index, is_relative=False)
>>> forecaster.fit(y_train)  # fit the forecaster on train series
>>> y_pred = forecaster.predict(fh)  # predict the test series time stamps
>>> mean_absolute_percentage_error(y_test, y_pred)
0.11725953222644162
```

`aeon` has a rich functionality for forecasting, and supports a wide variety of use
cases. To find out more about forecasting in `aeon`, you can explore through the
extensive [user guide notebook](./examples/forecasting/forecasting.ipynb).

## Time Series Classification (TSC)

Classification generally uses numpy arrays to store time series. We recommend storing
time series for classification in 3D numpy arrays of shape `(n_cases, n_channels,
n_timepoints)` even if each time series is univariate (i.e. `n_channels == 1`).
Classifiers will work with 2D input of shape `(n_cases, n_timepoints)` as you would
expect from `scikit-learn`, but other packages may treat 2D input as a single
multivariate series. This is the case for non-collection transformers, and you may
find unexpected outputs if you input a 2D array treating it as multiple time series.

Note we assume series length is always the same for all channels of a single series
regardless of input type. The target variable should be a `numpy` array of type `float`,
`int` or `str`.

The classification estimator interface should be familiar if you have worked with
`scikit-learn`. In this example we fit a [KNeighborsTimeSeriesClassifier](classification.distance_based.KNeighborsTimeSeriesClassifier)
with dynamic time warping (dtw) on our example data.

```{code-block} python
>>> import numpy as np
>>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
>>> X = [[[1, 2, 3, 4, 5, 6, 7]],  # 3D array example (univariate)
...      [[4, 4, 4, 5, 6, 7, 3]]]  # Two samples, one channel, seven series length
>>> y = [0, 1]  # class labels for each sample
>>> X = np.array(X)
>>> y = np.array(y)
>>> clf = KNeighborsTimeSeriesClassifier(distance="dtw")
>>> clf.fit(X, y)  # fit the classifier on train data
KNeighborsTimeSeriesClassifier()
>>> X_test = np.array([[2, 2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4, 4]])
>>> clf.predict(X_test)  # make class predictions on new data
[0 1]
```

Once the classifier has been fit using the training data and class labels, we can
predict the labels for new cases. Like `scikit-learn`, `predict_proba` methods are
available to predict class probabilities and a `score` method is present to
calculate accuracy on new data.

All `aeon` classifiers can be used with `scikit-learn` functionality for e.g.
model evaluation, parameter searching and pipelines. Explore the wide range of
algorithm types available in `aeon` in the [classification notebooks](examples.md#classification).

## Time Series Extrinsic Regression (TSER)

Time series extrinsic regression assumes that the target variable is continuous rather
than discrete, as for classification. The same input data considerations apply from the
classification section, and the modules function similarly. The target variable
should be a `numpy` array of type `float`.

"Time series regression" is a term commonly used in forecasting. To avoid confusion,
the term "time series extrinsic regression" is commonly used to refer to the traditional
machine learning regression task but for time series data.

In the following example we use a [KNeighborsTimeSeriesRegressor](regression.distance_based.KNeighborsTimeSeriesRegressor)
on an example time series extrinsic regression problem called [Covid3Month](https://zenodo.org/record/3902690).

```{code-block} python
>>> from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
>>> from aeon.datasets import load_covid_3month
>>> from sklearn.metrics import mean_squared_error
>>> X_train, y_train = load_covid_3month(split="train")
>>> X_test, y_test = load_covid_3month(split="test")
>>> reg = KNeighborsTimeSeriesRegressor(distance="dtw")
>>> reg.fit(X_train, y_train)  # fit the regressor on train data
KNeighborsTimeSeriesRegressor()
>>> y_pred = reg.predict(X_test)  # make label predictions on new data
>>> y_pred[:6]
[0.04218472 0.01459854 0.         0.0164468  0.06254257 0.11111111]
>>> mean_squared_error(y_test, y_pred)
0.002921957478363366
```

## Time Series Clustering (TSCL)

Like classification and regression, time series clustering aims to follow the
`scikit-learn` interface where possible. The same input data format is used as in
the TSC and TSER modules. This example fits a [TimeSeriesKMeans](clustering._k_means.TimeSeriesKMeans)
clusterer on the
[ArrowHead](http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead)
dataset.

```{code-block} python
>>> from aeon.clustering import TimeSeriesKMeans
>>> from aeon.datasets import load_arrow_head
>>> from sklearn.metrics import rand_score
>>> X, y = load_arrow_head()
>>> kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw")
>>> kmeans.fit(X) # fit the clusterer
TimeSeriesKMeans(n_clusters=3)
>>> kmeans.labels_[0:10]  # cluster labels
[2 1 1 0 1 1 0 1 1 0]
>>> rand_score(y, kmeans.labels_)
0.6377792823290453
```

After calling `fit`, the `labels_` attribute contains the cluster labels for
each time series. The `predict` method can be used to predict the cluster labels for
new data.

## Time Series Annotation

Annotation encompasses a range of time series tasks, including segmentation and anomaly
detection. The package is still in early development, so major changes are expected
as time goes on.

```{code-block} python
>>> from aeon.annotation.adapters import PyODAnnotator
>>> from aeon.datasets import load_airline
>>> from pyod.models.iforest import IForest
>>> y = load_airline()
>>> pyod_model = IForest()
>>> annotator = PyODAnnotator(pyod_model)
>>> annotator.fit(y)
>>> annotated_series = annotator.predict(y)
>>> annotated_series.head()
1949-01    1
1949-02    0
1949-03    0
1949-04    0
1949-05    0
Freq: M, dtype: int32
```

## Transformers for Time Series Data

The transformations module in `aeon` contains a range of transformers for time series
data. These transformers can be used standalone or as parts of pipelines.

Transformers inheriting from the [BaseTransformer](transformations.base.BaseTransformer)
class accept both series and collection input types. However, 2D input types can be
ambiguous in what they are storing. As such, we give the following warning for
`aeon` transformers: <span style="color:red">**2D input data types such as numpy arrays
and dataframes will be treated as a single multivariate series rather than a collection
of univariate series**</span>.


The following example shows how to use the
[Differencer](transformations.difference.Differencer) class to extract the first
order difference of a time series. Usage is the same for both single series and
collection input types.

```{code-block} python
>>> from aeon.transformations.difference import Differencer
>>> from aeon.datasets import load_airline
>>> from aeon.datasets import load_italy_power_demand
>>> diff = Differencer(lags=1)
>>> y = load_airline()  # load single series airline dataset
>>> diff.fit_transform(y)
Period
1949-01     0.0
1949-02     6.0
1949-03    14.0
1949-04    -3.0
1949-05    -8.0
           ...
1960-08   -16.0
1960-09   -98.0
1960-10   -47.0
1960-11   -71.0
1960-12    42.0
Freq: M, Name: Number of airline passengers, Length: 144, dtype: float64
>>> X, _ = load_italy_power_demand()  # load panel italy power demand dataset
>>> diff.fit_transform(X)[:2]
[[[ 0.         -0.47280283 -0.1891212  -0.2206413   0.1260808
    0.0945605   0.2836817   1.13472685  0.88256528  0.15760097
    0.1891211  -0.31520188 -0.34672208 -0.59888358 -0.66192396
    0.37824226  0.06304038  0.8195249   0.75648456  0.0945605
   -0.4097624  -0.47280285 -0.40976245 -0.44128264]]
 [[ 0.         -0.43377715 -0.1530978  -0.0255163  -0.0255163
    0.255163    0.3572282   0.66342387  1.07168459  0.48480974
   -0.0765489  -0.0765489  -0.25516304 -0.33171189  0.0255163
    0.0765489   0.0510326  -0.3061956  -0.05103261  0.84203794
   -0.0510326  -0.35722823 -0.73997271 -0.33171189]]]
```

As well as series-to-series transformations, the transformations module also contains
features which transform series into a feature vector. The following example shows how
to use the [SummaryTransformer](transformations.summarize.SummaryTransformer)
class to extract summary statistics from a time series.

```{code-block} python
>>> from aeon.transformations.summarize import SummaryTransformer
>>> from aeon.datasets import load_airline
>>> y = load_airline()  # load single series airline dataset
>>> summary = SummaryTransformer()
>>> summary.fit_transform(y)
         mean         std    min    max    0.1   0.25    0.5   0.75    0.9
0  280.298611  119.966317  104.0  622.0  135.3  180.0  265.5  360.5  453.2
```

## Transformers for Collections of Time Series

The `aeon.transformations.collections` module contains a range of transformers for
collections of time series. By default these do not allow for single series input,
treat 2D input types as a collection of univariate series, and have no restrictions on
the datatype of output.

Most time series classification and regression algorithms are based on some form of
transformation into an alternative feature space. For example, we might extract some
summary time series features from each series, and fit a traditional classifier or
regressor on these features. For example, we could use
[Catch22](transformations.collection.feauture_based), which calculates 22 summary
statistics for each series.

```{code-block} python
>>> from aeon.transformations.collection.feature_based import Catch22
>>> import numpy as np
>>> X = np.random.RandomState().random(size=(4, 1, 10))  # four cases of 10 timepoints
>>> c22 = Catch22(replace_nans=True)  # transform to four cases of 22 features
>>> c22.fit_transform(X)[0]
[ 4.99485761e-01  4.12452579e-01  3.00000000e+00  1.00000000e-01
  0.00000000e+00  1.00000000e+00  2.00000000e+00  3.08148791e-34
  1.96349541e+00  2.56152262e-01 -1.09028518e-02  9.08908735e-01
  2.00000000e+00  1.00000000e+00  4.00000000e+00  1.88915916e+00
  1.00000000e+00  5.95334611e-01  0.00000000e+00  0.00000000e+00
  8.23045267e-03  0.00000000e+00]
```

There are also series-to-series transformations, such as the
[PaddingTransformer](transformations.collection.pad.PaddingTransformer) to lengthen
series and process unequal length collections.

```{code-block} python
>>> from aeon.transformations.collection.pad import PaddingTransformer
>>> from aeon.testing.utils.data_gen import make_example_unequal_length
>>> X, _ = make_example_unequal_length(  # unequal length data with 8-12 timepoints
...     n_cases=2,
...     min_n_timepoints=8,
...     max_n_timepoints=12,
...     random_state=0,
... )
>>> print(X[0])
[[0.         1.6885315  1.71589124 1.69450348 1.24712739 0.76876341
  0.59506921 0.11342595 0.54531259 0.95533023 1.62433746 0.95995434]]
>>> print(X[1])
[[2.         0.28414423 0.3485172  0.08087359 3.33047938 3.112627
  3.48004859 3.91447337 3.19663426]]
>>> pad = PaddingTransformer(pad_length=12, fill_value=0)  # pad to length 12
>>> pad.fit_transform(X)
[[[0.         1.6885315  1.71589124 1.69450348 1.24712739 0.76876341
   0.59506921 0.11342595 0.54531259 0.95533023 1.62433746 0.95995434]]
 [[2.         0.28414423 0.3485172  0.08087359 3.33047938 3.112627
   3.48004859 3.91447337 3.19663426 0.         0.         0.        ]]]
```

If single series input is required, regular transformer functionality can be restored
using the
[CollectionToSeriesWrapper](transformations.collection.CollectionToSeriesWrapper) class.
Like other `BaseTransformer` classes, this wrapper will treat 2D input as a single
multivariate series and automatically convert output.

```{code-block} python
>>> from aeon.transformations.collection import CollectionToSeriesWrapper
>>> from aeon.transformations.collection.feature_based import Catch22
>>> from aeon.datasets import load_airline
>>> y = load_airline()  # load single series airline dataset
>>> c22 = Catch22(replace_nans=True)
>>> wrapper = CollectionToSeriesWrapper(c22)  # wrap transformer to accept single series
>>> wrapper.fit_transform(y)
           0           1     2         3   ...        18        19        20    21
0  155.800003  181.700012  49.0  0.541667  ...  0.282051  0.769231  0.166667  11.0

[1 rows x 22 columns]
```

## Pipelines for aeon estimators

Like `scikit-learn`, `aeon` provides pipeline classes which can be used to chain
transformations and estimators together. The simplest pipeline for forecasting is the
[TransformedTargetForecaster](forecasting.compose.TransformedTargetForecaster).

In the following example, we chain together a
[BoxCoxTransformer](transformations.boxcox.BoxCoxTransformer),
[Deseasonalizer](transformations.detrend.Deseasonalizer) and
[ARIMA](forecasting.arima.ARIMA) forecaster to make a forecast (if you want to run this
yourself, you will need to `pip install statsmodels` and `pip install pmdarima`).

```{code-block} python
>>> import numpy as np
>>> from aeon.datasets import load_airline
>>> from aeon.transformations.boxcox import BoxCoxTransformer
>>> from aeon.transformations.detrend import Deseasonalizer
>>> from aeon.forecasting.arima import ARIMA
>>> from aeon.forecasting.compose import TransformedTargetForecaster
...
>>> # Load airline data
>>> y = load_airline()
>>> # Create and fit the pipeline
>>> pipe = TransformedTargetForecaster(
...     steps=[
...         ("boxcox", BoxCoxTransformer(sp=12)),
...         ("deseasonaliser", Deseasonalizer(sp=12)),
...         ("arima", ARIMA(order=(1, 1, 0))),
...     ]
... )
>>> pipe.fit(y)
>>> # Make predictions
>>> pipe.predict(fh=np.arange(1, 13))
1961-01    442.440026
1961-02    433.548016
1961-03    493.371215
1961-04    484.284090
1961-05    490.850617
1961-06    555.134680
1961-07    609.581248
1961-08    611.345923
1961-09    542.610868
1961-10    482.452172
1961-11    428.885045
1961-12    479.297989
Freq: M, dtype: float64
```

For most learning tasks including forecasting, the `aeon` [make_pipeline](pipeline.make_pipeline)
function can be used to creating pipelines as well.

```{code-block} python
>>> from aeon.pipeline import make_pipeline
>>> make_pipeline(
...     BoxCoxTransformer(sp=12), Deseasonalizer(sp=12), ARIMA(order=(1, 1, 0))
... )
TransformedTargetForecaster(steps=[BoxCoxTransformer(sp=12),
                                   Deseasonalizer(sp=12),
                                   ARIMA(order=(1, 1, 0))])
```

For machine learning tasks such as classification, regression and clustering, the
`scikit-learn` `make_pipeline` functionality can be used if the transformer outputs
a valid input type.

The following example uses the [Catch22](transformations.collection.catch22.Catch22)
feature extraction transformer and a random forest classifier to classify.

```{code-block} python
>>> from aeon.datasets import load_italy_power_demand
>>> from aeon.transformations.collection.feature_based import Catch22
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.metrics import accuracy_score
...
>>> # Load the italy power demand dataset
>>> X_train, y_train = load_italy_power_demand(split="train")
>>> X_test, y_test = load_italy_power_demand(split="test")
...
>>> # Create and fit the pipeline
>>> pipe = make_pipeline(
...     Catch22(replace_nans=True),
...     RandomForestClassifier(random_state=42),
... )
>>> pipe.fit(X_train, y_train)
Pipeline(steps=[('catch22', Catch22(replace_nans=True)),
                ('randomforestclassifier',
                 RandomForestClassifier(random_state=42))])
>>> # Make predictions like any other sklearn estimator
>>> accuracy_score(pipe.predict(X_test), y_test)
0.8989310009718173
```

## Parameter searching for aeon estimators

Tools for selecting parameter values for `aeon` estimators are available. In the
following example, we use a [ForecastingGridSearchCV](forecasting.model_selection.ForecastingGridSearchCV)
to ARIMA order values for the forecasting pipeline we created in the previous example.

```{code-block} python
>>> import warnings
>>> import numpy as np
>>> from itertools import product
>>> from sklearn.exceptions import ConvergenceWarning
>>> from aeon.datasets import load_airline
>>> from aeon.forecasting.compose import TransformedTargetForecaster
>>> from aeon.forecasting.model_selection import (
...     ExpandingWindowSplitter,
...     ForecastingGridSearchCV,
... )
>>> from aeon.forecasting.arima import ARIMA
>>> from aeon.transformations.boxcox import BoxCoxTransformer
>>> from aeon.transformations.detrend import Deseasonalizer
...
>>> y = load_airline()
...
>>> cv = ExpandingWindowSplitter(initial_window=120, fh=np.arange(1, 13))
>>> arima_orders = list(product((0, 1, 2), (0, 1, 2), (0, 1, 2)))
...
>>> warnings.simplefilter("ignore", category=ConvergenceWarning)
>>> gscv = ForecastingGridSearchCV(
...     forecaster=TransformedTargetForecaster(
...         steps=[
...             ("boxcox", BoxCoxTransformer(sp=12)),
...             ("deseasonaliser", Deseasonalizer(sp=12)),
...             ("arima", ARIMA(order=(1, 1, 0))),
...        ]
...     ),
...     param_grid={"arima__order": arima_orders},
...     cv=cv,
... )
>>> gscv.fit(y)
...
>>> gscv.predict(fh=np.arange(1, 13))
1961-01    443.073816
1961-02    434.309107
1961-03    494.198070
1961-04    485.105623
1961-05    491.684116
1961-06    556.064082
1961-07    610.591655
1961-08    612.362761
1961-09    543.533022
1961-10    483.289701
1961-11    429.645587
1961-12    480.137248
Freq: M, dtype: float64
>>> gscv.best_params_["arima__order"]
(0, 1, 1)
```

Like with pipelines, tasks such as classification, regression and clustering can use
the available `scikit-learn` functionality.

```{code-block} python
>>> from sklearn.metrics import accuracy_score
>>> from sklearn.model_selection import GridSearchCV, KFold
>>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
>>> from aeon.datasets import load_italy_power_demand
...
>>> # Load the italy power demand dataset
>>> X_train, y_train = load_italy_power_demand(split="train")
>>> X_test, y_test = load_italy_power_demand(split="test")
...
>>> knn = KNeighborsTimeSeriesClassifier()
>>> param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}
...
>>> gscv = GridSearchCV(knn, param_grid, cv=KFold(n_splits=4))
>>> gscv.fit(X_train, y_train)
...
>>> y_pred = gscv.predict(X_test)
>>> y_pred[:6]
['2' '2' '2' '2' '2' '1']
>>> accuracy_score(y_test, y_pred)
0.9523809523809523
>>> gscv.best_params_
{'distance': 'euclidean', 'n_neighbors': 5}
```

## Time series similarity search

The similarity search module in `aeon` offers a set of functions and estimators to solve
tasks related to time series similarity search. The estimators can be used standalone
or as parts of pipelines, while the functions give you the tools to build your own
estimators that would rely on similarity search at some point.

The estimators are inheriting from the [BaseSimiliaritySearch](similarity_search.base.BaseSimiliaritySearch)
class accepts as inputs 3D time series (n_cases, n_channels, n_timepoints) for the
fit method. Univariate and single series can still be used, but will need to be reshaped
to this format.

This collection, asked for the fit method, is stored as a database. It will be used in
the predict method, which expects a single 2D time series as input
(n_channels, query_length), which will be used as a query to search for in the database.
Note that the length of the time series in the 3D  collection should be superior or
equal to the length of the 2D time series given in the predict method.

Given those two inputs, the predict method should return the set of most similar
candidates to the 2D series in the 3D collection. The following example shows how to use
the [TopKSimilaritySearch](similarity_search.top_k_similarity.TopKSimilaritySearch)
class to extract the best `k` matches, using the Euclidean distance as similarity
function.

```{code-block} python
>>> import numpy as np
>>> from aeon.similarity_search import TopKSimilaritySearch
>>> X = [[[1, 2, 3, 4, 5, 6, 7]],  # 3D array example (univariate)
...      [[4, 4, 4, 5, 6, 7, 3]]]  # Two samples, one channel, seven series length
>>> X = np.array(X) # X is of shape (2, 1, 7) : (n_cases, n_channels, n_timepoints)
>>> topk = TopKSimilaritySearch(distance="euclidean",k=2)
>>> topk.fit(X)  # fit the estimator on train data
...
>>> q = np.array([[4, 5, 6]]) # q is of shape (1,3) :
>>> topk.predict(q)  # Identify the two (k=2) most similar subsequences of length 3 in X
[(0, 3), (1, 2)]
```
The output of predict gives a list of size `k`, where each element is a set indicating
the location of the best matches in X as `(id_sample, id_timestamp)`. This is equivalent
to the subsequence `X[id_sample, :, id_timestamps:id_timestamp + q.shape[0]]`.

Note that you can still use univariate time series as inputs, you will just have to
convert them to multivariate time series with one feature prior to using the similarity
search module.
