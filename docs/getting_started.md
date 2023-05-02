# Getting Started

The following information is designed to get users up and running with `aeon` quickly.
If installation is required, please see our [installation guide](installation) for
installing `aeon`.

We assume basic familiarity with [scikit-learn](https://scikit-learn.org/stable/index.html)
package. If you are confused, you may want to view
[their getting started guides](https://scikit-learn.org/stable/getting_started.html).

`aeon` is an open source toolkit for learning from time series. It provides access to
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

Additionally, it provides numerous algorithms for {term}`time series transformation`,
altering time series into different representations and domains or processing
time series data into tabular data.

The following provides introductory examples for each of these modules. In the examples
used the datatypes most commonly used for the task, but a variety of input types for
data are available. See [here](/examples/datasets/AA_datatypes_and_datasets.ipynb) for
more information on input datatypes. For more information on the variety of estimators
available for each task, see the [API](api_reference) and [examples](examples) pages.

## Forecasting

Forecasting primarily uses pandas DataFrames and Series objects to store time series
data. An example of data in thie format is the Airline dataset.

```{code-block} python
>>> from aeon.datasets import load_airline
>>> y = load_airline()  # load an example series with timestamps
>>> type(y)
<class 'pandas.core.series.Series'>
>>> y.head()
Period
1960-08    606.0
1960-09    508.0
1960-10    461.0
1960-11    390.0
1960-12    432.0
Freq: M, Name: Number of airline passengers, dtype: float64
```

The possible use cases for forecasting are more complex than with the other modules.
Like `scikit-learn`, forecasters use a fit and predict model, but the arguments are
different. The simplest forecasting use case is when you have a single series and you
want to build a model on that series (e.g. ARMA model) to predict values in the
future. At their most basic, forecasters require a series to forecast for fit, and a
forecast horizon (`fh`) to specify how many time steps ahead to make a forecast in
predict. This code fits a [TrendForecaster](forecasting.trend.TrendForecaster) on our
loaded data and predicts the next value in the series.

```{code-block} python
>>> from aeon.forecasting.trend import TrendForecaster
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

In this getting started tutorial we assume a single series univariate input. `aeon`
has a rich functionality for forecasting, and supports a wide variety of use
cases. To find out more about forecasting in `aeon`, you can explore through the
extensive [user guide notebook](./examples/forecasting/forecasting.ipynb).

## Time Series Classification (TSC)

Classification generally use numpy arrays to store time series. For 2D input, the size
should be (n_cases, series_length) and for 3D the size should be (n_cases, n_channels,
series_length). Each case in our input data set is a time series.

```{code-block} python
>>> X = [[1, 2, 3, 4, 5, 6, 7],  # 2D array example (univariate)
...      [4, 4, 4, 5, 6, 7, 3]]  # Two samples, one channel, seven series length
>>> X2 = [[[1, 2, 3, 4], [3, 8, 3, 8]],  # 3D array example (multivariate)
...       [[5, 2, 1, 5], [3, 8, 3, 8]]]  # Two samples, two channels, four series length
>>> y = [0, 1]  # class labels for each sample
```

So, for example, `X1[0]` is a univariate time series of length seven, and `X2[0]` is a
multivariate time series of length four which has three channels.

By default, we recommend storing time series for classification in 3D numpy arrays even
if each time series is univariate. Classifiers will work with 2D input, but other
packages may treat 2D input as a single multivariate series. This is the case for
non-collection transformers, and you may find unexpected outputs if you input a 2D array
treating it as multiple time series.

The classification estimator interface should be familiar if you have worked with
`scikit-learn`. In this example we fit a [KNeighborsTimeSeriesClassifier](classification.distance_based.KNeighborsTimeSeriesClassifier)
on our example data.

```{code-block} python
>>> import numpy as np
>>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
>>> X = np.array(X)
>>> y = np.array(y)
>>> clf = KNeighborsTimeSeriesClassifier(distance="dtw")
>>> clf.fit(X, y)  # fit the classifier on train data
KNeighborsTimeSeriesClassifier()
>>> X_test = np.array([[2, 2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4, 4]])
>>> y_pred = clf.predict(X_test)  # make class predictions on new data
[0 1]
```

Once the classifier has been fit using the training data and class labels, we can
predict the labels for new cases. Like `scikit-learn`, `predict_proba` methods are
available to predict class probabilities and a `score` method is present to
calculate accuracy on new data.

All `aeon` classifiers can be used with `scikit-learn` functionality for e.g.
model evaluation, parameter searching and pipelines. Explore the wide range of
algorithm types available in ``aeon` in the [classification notebooks](examples.md#classification).

## Time Series Extrinsic Regression (TSER)

Time series extrinsic regression assumes that the target variable is continuous rather
than discrete, as for classification. The same input data considerations apply from the
classification section, and the modules function similarly.

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
`scikit-learn` interface where possible. The same input data format is used as the
modules. This example fits a [TimeSeriesKMeans](clustering.k_means.TimeSeriesKMeans)
clusterer on the [ArrowHead](http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead)
dataset.

```{code-block} python
>>> from aeon.clustering.k_means import TimeSeriesKMeans
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
