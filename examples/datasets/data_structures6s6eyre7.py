# %%NBQA-CELL-SEPfc780c
# Forecasting data in a pandas.Series
import numpy as np
import pandas as pd

from aeon.forecasting.trend import TrendForecaster

y = pd.Series([20.0, 40.0, 60.0, 80.0, 100.0])
forecaster = TrendForecaster()
forecaster.fit(y)  # fit the forecaster
forecaster.predict(fh=[1, 2, 3])  # forecast the next 3 values


# %%NBQA-CELL-SEPfc780c
ice_creams = {
    "Sales": [111, 100, 90, 80, 65, 89],
    "Temperature": [26, 21, 19, 14, 12, 22],
}
# Create DataFrame
ice_creams = pd.DataFrame(ice_creams)
print(ice_creams)
from aeon.forecasting.exp_smoothing import ExponentialSmoothing

forecaster = ExponentialSmoothing()
forecaster.fit(ice_creams)
forecaster.predict(fh=[1, 2, 3])


# %%NBQA-CELL-SEPfc780c
ice_creams["datetime"] = pd.to_datetime(
    [
        "01-06-2018 23:15:00",  # Creating data
        "02-09-2019 01:48:00",
        "08-06-2020 13:20:00",
        "07-03-2021 14:50:00",
        "07-06-2022 11:50:00",
        "03-05-2023 16:50:00",
    ]
)
ice_creams = ice_creams.set_index("datetime")
print(ice_creams)


# %%NBQA-CELL-SEPfc780c
from aeon.testing.data_generation import _make_hierarchical

y = _make_hierarchical()
y.head()


# %%NBQA-CELL-SEPfc780c
forecaster.fit(y, fh=[1, 2]).predict()


# %%NBQA-CELL-SEPfc780c
y = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
forecaster = TrendForecaster()
forecaster.fit(y)  # fit the forecaster
forecaster.predict(fh=[1, 2, 3])  # forecast the next 3 values


# %%NBQA-CELL-SEPfc780c
y = np.array([[20.0, 40.0, 60.0, 80.0, 100.0], [100.0, 90.0, 80.0, 70.0, 60.0]])
y = y.transpose()
forecaster = TrendForecaster()
forecaster.fit(y)  # fit the forecaster
forecaster.predict(fh=[1, 2, 3])  # forecast the next 3 values


# %%NBQA-CELL-SEPfc780c
X = np.array(
    [
        [[20.0, 40.0, 60.0, 80.0, 100.0]],  # Univariate series as 3D array
        [[100.0, 90.0, 80.0, 70.0, 60.0]],
    ]
)  # n_cases = 2, n_channels =1, n_timepoints = 5
print("X shape = ", X.shape, " First series =", X[0], "second series = ", X[1])


# %%NBQA-CELL-SEPfc780c
X = np.array(
    [
        [[20, 40, 600, 55], [10, 11, 12, 11], [-4, 1, 6.6, 2]],
        [[10, 90, 80, 100], [14, 70, 60, 22], [49, 49, 66, 9]],
        [[14, 6, 10, -401], [44, 70, 60, 22], [49, 52, 33, 49]],
        [[22, 93, 18, 100], [34, 170, 0, 87], [49, 49, 33, 49]],
    ]
)
# n_cases = 4, n_channels =3, n_timepoints = 4
print("X shape = ", X.shape, "\n First series =\n", X[0], "\nsecond series = \n", X[1])
from aeon.clustering import TimeSeriesKMeans

kmeans = TimeSeriesKMeans(distance="euclidean", n_clusters=2)
kmeans.fit(X)
kmeans.predict(X)


# %%NBQA-CELL-SEPfc780c
y = np.array([1, 1, 0, 0])
y2 = np.array(["pass", "pass", "fail", "fail"])
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

knn = KNeighborsTimeSeriesClassifier(distance="dtw")
knn.fit(X, y)
knn.fit(X, y2)
knn.predict(X)


# %%NBQA-CELL-SEPfc780c
y = np.array([1.5, 4.3, -2.0, 10])
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

knn_r = KNeighborsTimeSeriesRegressor(distance="dtw")
knn_r.fit(X, y)
knn_r.predict(X)


# %%NBQA-CELL-SEPfc780c
x0 = np.array([[20, 40, 60, 55, 66], [10, 11, 12, 11, 66], [-4, 15, 6.6, 12, 44]])
x1 = np.array([[10, 90, 80], [70, 60, 22], [49, 66, 9]])
x2 = np.array([[22, 93, 18, 100], [34, 170, 0, 87], [49, 49, 33, 49]])
X_uneq = []
X_uneq.append(x0)
X_uneq.append(x1)
X_uneq.append(x2)
y = np.array([0, 0, 1])
knn.fit(X_uneq, y)
knn.predict(X_uneq)
