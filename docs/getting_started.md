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

- {term}`Time series classification` where the time series data for a given instance
are used to predict a categorical target class.
- {term}`Time series extrinsic regression` where the time series data for a given
instance are used to predict a continuous target value.
- {term}`Time series clustering` where the goal is to discover groups consisting of
instances with similar time series.
- {term}`Time series similarity search` where the goal is to evaluate the similarity
between a time series against a collection of other time series.

Additionally, it provides numerous algorithms for {term}`time series transformation`,
altering time series into different representations and domains or processing
time series data into tabular data.

The following provides introductory examples for each of these modules. The examples
use the datatypes most commonly used for the task in question, but a variety of input
types for
data are available. For more information on the variety of
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

We commonly refer to the number of observations for a time series as `n_timepoints`. If a series is multivariate, we refer to the dimensions as channels
(to avoid confusion with the dimensions of array) and in code use `n_channels`.
Dimensions may also be referred to as variables.

Different parts of `aeon` work with single series or collections of series. The
`anomaly detection` and `segmentation` modules will commonly use single series input, while
`classification`, `regression` and `clustering` modules will use collections of time
series. Collections of time series may also be referred to as Panels. Collections of
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


## Transformers for Single Time Series

Transformers inheriting from the [BaseSeriesTransformer](transformations.base.BaseSeriesTransformer)
in the `aeon.transformations.series` transform a single (possibly multivariate) time
series into a different time series or a feature vector.

The following example shows how to use the
[AutoCorrelationSeriesTransformer](transformations.series.AutoCorrelationSeriesTransformer)
class to extract the autocorrelation terms of a time series.

```{code-block} python
>>> from aeon.transformations.series import AutoCorrelationSeriesTransformer
>>> from aeon.datasets import load_airline
>>> acf = AutoCorrelationSeriesTransformer()
>>> y = load_airline()  # load single series airline dataset
>>> res = acf.fit_transform(y)
>>> res[0][:5]
[0.96019465 0.89567531 0.83739477 0.7977347  0.78594315]
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
[Padder](transformations.collection) to lengthen
series and process unequal length collections.

```{code-block} python
>>> from aeon.transformations.collection import Padder
>>> from aeon.testing.data_generation import make_example_3d_numpy_list
>>> X, _ = make_example_3d_numpy_list(  # unequal length data with 8-12 timepoints
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
>>> pad = Padder(pad_length=12, fill_value=0)  # pad to length 12
>>> pad.fit_transform(X)
[[[0.         1.6885315  1.71589124 1.69450348 1.24712739 0.76876341
   0.59506921 0.11342595 0.54531259 0.95533023 1.62433746 0.95995434]]
 [[2.         0.28414423 0.3485172  0.08087359 3.33047938 3.112627
   3.48004859 3.91447337 3.19663426 0.         0.         0.        ]]]
```


## Pipelines for aeon estimators

Like `scikit-learn`, `aeon` provides pipeline classes which can be used to chain
transformations and estimators together.

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
