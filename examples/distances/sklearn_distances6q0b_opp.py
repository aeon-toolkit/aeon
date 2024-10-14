# %%NBQA-CELL-SEPfc780c
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer

from aeon.datasets import load_gunpoint

# Load the gunpoint dataset as a 2D numpy array
X_train_2D, y_train_2D = load_gunpoint(split="TRAIN", return_type="numpy2D")
X_test_2D, y_test_2D = load_gunpoint(split="TEST", return_type="numpy2D")

# Load the gunpoint dataset as a 3D numpy array
X_train_3D, y_train_3D = load_gunpoint(split="TRAIN")
X_test_3D, y_test_3D = load_gunpoint(split="TEST")


# %%NBQA-CELL-SEPfc780c
# Using the 2D array format
print(f"Training set shape = {X_train_2D.shape} -> this works with sklearn")

# Apply a sklearn kNN classifier on the 2D
#  time series data using a standard distance
knn = KNeighborsClassifier(metric="manhattan")
knn.fit(X_train_2D, y_train_2D)
predictions_2D = knn.predict(X_test_2D[:5])
print(f"kNN with manhattan distance on 2D time series data {predictions_2D}\n")


# Now using the 3D array format
print(f"Training set shape = {X_train_3D.shape} -> sklearn will crash as is a 3D array")

# Apply a sklearn kNN classifier on the 3D time series data using a standard distance
# This will raise a ValueError as sklearn does not support 3D arrays
try:
    knn.fit(X_train_3D, y_train_3D)
except ValueError as e:
    print(f"Raises this ValueError:\n\t{e}")


# %%NBQA-CELL-SEPfc780c
from aeon.distances import (
    adtw_distance,
    dtw_distance,
    edr_distance,
    msm_distance,
    twe_distance,
)

# Apply a sklearn kNN classifier on the 2D time series data using the DTW distance
knn = KNeighborsClassifier(metric=dtw_distance)
knn.fit(X_train_2D, y_train_2D)
predictions_2D_DTW = knn.predict(X_test_2D[:5])

print(f"kNN with DTW distance on 2D time series data {predictions_2D_DTW}\n")


# Apply a sklearn kNN classifier on the 3D time series data using the DTW distance
# This will still raise a ValueError as sklearn does not support 3D arrays
print("kNN with DTW distance on 3D time series data...")
try:
    knn.fit(X_train_3D, y_train_3D)
except ValueError as e:
    print(f"...raises this ValueError:\n\t{e}")


# %%NBQA-CELL-SEPfc780c
# Transform X into a graph of k nearest neighbors on the 2D time series data using the
# EDR distance
kt = KNeighborsTransformer(
    metric=edr_distance,
    metric_params={"itakura_max_slope": 0.5},
)

kt.fit(X_train_2D, y_train_2D)
kgraph = kt.transform(X_test_2D[:1]).toarray()  # Convert the sparse matrix to an array

print(
    "Graph of neighbors for the first pattern in testing set with EDR distance on 2D"
    f"time series data:\n{kgraph}\nNote that [i,j] has the weight of edge that "
    "connects i to j.\n"
)

# Again, using a 3D array will raise a ValueError
print("Again, transforming 3D time series data into a graph of neighbors...")
try:
    kt.fit(X_train_3D, y_train_3D)
except ValueError as e:
    print(f"...raises this ValueError:\n\t{e}")


# %%NBQA-CELL-SEPfc780c
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

# Apply the aeon kNN classifier on the 3D time series data using the MSM distance
knn_aeon = KNeighborsTimeSeriesClassifier(distance="msm")
knn_aeon.fit(X_train_3D, y_train_3D)

predictions_3D_aeon = knn_aeon.predict(X_test_3D[:5])

print(f"aeon kNN with MSM distance on 3D time series data {predictions_3D_aeon}")

# Apply a sklearn kNN classifier on the 2D time series data using the MSM distance
knn = KNeighborsClassifier(metric=msm_distance)
knn.fit(X_train_2D, y_train_2D)
predictions_2D_sklearn = knn.predict(X_test_2D[:5])

print(f"sklearn kNN with MSM distance on 2D time series data {predictions_2D_sklearn}")


# %%NBQA-CELL-SEPfc780c
import numpy as np

from aeon.datasets import load_basic_motions

# Load the basic_motions multivariate (MTSC) dataset as a 3D numpy array
X_train_3D_mtsc, y_train_mtsc = load_basic_motions(split="TRAIN")
X_test_3D_mtsc, y_test_mtsc = load_basic_motions(split="TEST")

print(f"3D training set shape = {X_train_3D_mtsc.shape} -> does not work with sklearn")

# Transform the 3D numpy array to 2D concatenating the time series
# This time, the loader does not return the dataset as a 2D array as this is not an
# intended way of working with time series. We need to reshape it ourselves.
X_train_2D_mtsc = X_train_3D_mtsc.reshape(X_train_3D_mtsc.shape[0], -1)
X_test_2D_mtsc = X_test_3D_mtsc.reshape(X_test_3D_mtsc.shape[0], -1)

print(f"2D Training set shape = {X_train_2D_mtsc.shape} -> works with sklearn")

# selects some patterns from the dataset to speed up the example
indices = np.random.RandomState(1234).choice(len(y_test_mtsc), 5, replace=False)

X_test_2D_mtsc = X_test_2D_mtsc[indices]
X_test_3D_mtsc = X_test_3D_mtsc[indices]
y_test_mtsc = y_test_mtsc[indices]


# %%NBQA-CELL-SEPfc780c
# Apply the aeon kNN classifier on the 3D MTSC time series data using the ADTW distance
knn_aeon = KNeighborsTimeSeriesClassifier(distance="adtw")
knn_aeon.fit(X_train_3D_mtsc, y_train_mtsc)

predictions_3D_aeon = knn_aeon.predict(X_test_3D_mtsc)

print(f"aeon kNN with MSM distance on 3D MTSC time series data {predictions_3D_aeon}")

# Apply a sklearn kNN classifier on the concatenated 2D MTSC time series data using the
# ADTW distance
knn = KNeighborsClassifier(metric=adtw_distance)
knn.fit(X_train_2D_mtsc, y_train_mtsc)
predictions_2D_sk = knn.predict(X_test_2D_mtsc)

print(f"sklearn kNN with MSM distance on 2D MTSC time series data {predictions_2D_sk}")


# %%NBQA-CELL-SEPfc780c
from sklearn.neighbors import KNeighborsRegressor

from aeon.datasets import load_covid_3month
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

# Load the Covid3Month dataset as a 3D numpy array
X_train_3D_reg, y_train_3D_reg = load_covid_3month(split="train")
X_test_3D_reg, y_test_3D_reg = load_covid_3month(split="test")

# Load the Covid3Month dataset as a 2D numpy array
X_train_2D_reg, y_train_2D_reg = load_covid_3month(split="train", return_type="numpy2D")
X_test_2D_reg, y_test_2D_reg = load_covid_3month(split="test", return_type="numpy2D")


# %%NBQA-CELL-SEPfc780c
knn_aeon_reg = KNeighborsTimeSeriesRegressor(distance="twe", n_neighbors=1)
knn_aeon_reg.fit(X_train_3D_reg, y_train_3D_reg)

predictions_3D_reg_aeon = knn_aeon_reg.predict(X_test_3D_reg[:5])

print(
    f"aeon kNN with MSM distance on 3D TSER time series data {predictions_3D_reg_aeon}"
)

knn_sklearn = KNeighborsRegressor(metric=twe_distance, n_neighbors=1)
knn_sklearn.fit(X_train_2D_reg, y_train_2D_reg)

predictions_2D_reg_sk = knn_aeon_reg.predict(X_test_2D_reg[:5])

print(
    f"sklearn kNN with MSM distance on 2D TSER time series data {predictions_2D_reg_sk}"
)


# %%NBQA-CELL-SEPfc780c
from sklearn.metrics import accuracy_score

from aeon.distances import adtw_pairwise_distance

# Compute the distances between all pairs of time series in the training set
# and between the testing set and the training set for the testing set
train_dists = adtw_pairwise_distance(X_train_3D)
test_dists = adtw_pairwise_distance(X_test_3D, X_train_3D)

# scikit-learn KNN classifier with precomputed distances
knn = KNeighborsClassifier(metric="precomputed", n_neighbors=1)
knn.fit(train_dists, y_train_3D)
predictions_precomputed = knn.predict(test_dists)

print(f"sklearn kNN with precomputed ADTW distance {predictions_precomputed[:5]}")

# aeon KNN classifier with ADTW distance (not precomputed)
knn_aeon = KNeighborsTimeSeriesClassifier(distance="adtw", n_neighbors=1)
knn_aeon.fit(X_train_3D, y_train_3D)
predictions_aeon = knn_aeon.predict(X_test_3D)

print(f"aeon kNN with ADTW distance {predictions_aeon[:5]}")

# Compute the CCR on both experiments
CCR_precomputed = accuracy_score(y_test_3D, predictions_precomputed)
CCR_aeon = accuracy_score(y_test_3D, predictions_aeon)

print(f"{CCR_precomputed=}\n{CCR_aeon=}")


# %%NBQA-CELL-SEPfc780c
from sklearn.metrics import mean_squared_error

from aeon.distances import erp_pairwise_distance

# Compute the distances between all pairs of time series in the training set
# and between the testing set and the training set for the testing set
train_dists_erp = erp_pairwise_distance(X_train_3D_reg)
test_dists_erp = erp_pairwise_distance(X_test_3D_reg, X_train_3D_reg)

# scikit-learn KNN regressor with precomputed distances
knn = KNeighborsRegressor(metric="precomputed", n_neighbors=1)
knn.fit(train_dists_erp, y_train_3D_reg)
predictions_precomputed = knn.predict(test_dists_erp)

print(f"sklearn kNN with precomputed ERP distance {predictions_precomputed[:5]}")

# aeon KNN regressor with ERP distance (not precomputed)
knn_aeon = KNeighborsTimeSeriesRegressor(distance="erp", n_neighbors=1)
knn_aeon.fit(X_train_3D_reg, y_train_3D_reg)
predictions_aeon = knn_aeon.predict(X_test_3D_reg)

print(f"aeon kNN with ERP distance {predictions_aeon[:5]}")

# Compute the CCR on both experiments
MSE_precomputed = mean_squared_error(y_test_3D_reg, predictions_precomputed)
MSE_aeon = mean_squared_error(y_test_3D_reg, predictions_aeon)

print(f"{MSE_precomputed=}\n{MSE_aeon=}")


# %%NBQA-CELL-SEPfc780c
from sklearn.svm import SVC

from aeon.distances import twe_pairwise_distance

# %%NBQA-CELL-SEPfc780c
# Select 25 patterns from the dataset to speed up the example
indices = np.random.RandomState(1234).choice(len(y_train_3D), 25, replace=False)

# Fit the SVC model with the TWE distance as callable function.
svc = SVC(kernel=twe_pairwise_distance)
svc.fit(X_train_2D[indices], y_train_2D[indices])

print("SVC with TWE first five predictions = ", svc.predict(X_test_2D)[:5])


# %%NBQA-CELL-SEPfc780c
# Fit the SVC model with precomputed distances
svc = SVC(kernel="precomputed")
train_dists_twe = twe_pairwise_distance(X_train_3D[indices])
test_dists_twe = twe_pairwise_distance(X_test_3D, X_train_3D[indices])
svc.fit(train_dists_twe, y_train_3D[indices])

print(
    "SVC with precomputed TWE first five predictions = ",
    svc.predict(test_dists_twe)[:5],
)


# %%NBQA-CELL-SEPfc780c
from sklearn.svm import NuSVR

from aeon.distances import msm_pairwise_distance

# %%NBQA-CELL-SEPfc780c
# Select 25 patterns from the dataset to speed up the example
indices = np.random.RandomState(1234).choice(len(y_train_3D_reg), 25, replace=False)

# Fit the NuSVR model with the MSM distance as callable function.
nusvr = NuSVR(kernel=msm_pairwise_distance)
nusvr.fit(X_train_2D_reg[indices], y_train_2D_reg[indices])

print("NuSVR with MSM first five predictions = ", nusvr.predict(X_test_2D_reg)[:5])

# Fit the NuSVR model with precomputed distances
nusvr = NuSVR(kernel="precomputed")
train_dists_msm = msm_pairwise_distance(X_train_3D_reg[indices])
test_dists_msm = msm_pairwise_distance(X_test_3D_reg, X_train_3D_reg[indices])
nusvr.fit(train_dists_msm, y_train_3D_reg[indices])

print(
    "NuSVR with precomputed MSM first five predictions = ",
    nusvr.predict(test_dists_msm)[:5],
)


# %%NBQA-CELL-SEPfc780c
from sklearn.cluster import DBSCAN

from aeon.distances import euclidean_pairwise_distance

# %%NBQA-CELL-SEPfc780c
# Fit the DBSCAN model with the euclidean distance (default).
db1 = DBSCAN(eps=2.5)
preds1 = db1.fit_predict(X_train_2D)
print("DBSCAN with euclidean distance on 2D time series data = ", preds1[:5])

# Fit the DBSCAN model with precomputed distances
db2 = DBSCAN(metric="precomputed", eps=2.5)
preds2 = db2.fit_predict(euclidean_pairwise_distance(X_train_3D))
print(
    "DBSCAN with precomputed euclidean distance on 3D time series data = ",
    preds2[:5],
)


# Fit the DBSCAN model with the MSM distance as callable function.
db3 = DBSCAN(metric=msm_distance, eps=2.5)
preds3 = db3.fit_predict(X_train_2D)
print("DBSCAN with MSM distance on 2D time series data = ", preds3[:5])

# Fit the DBSCAN model with precomputed distances on the MSM distance
db4 = DBSCAN(metric="precomputed", eps=2.5)
preds4 = db4.fit_predict(msm_pairwise_distance(X_train_3D))
print(
    "DBSCAN with precomputed MSM distance on 3D time series data = ",
    preds4[:5],
)


# %%NBQA-CELL-SEPfc780c
from sklearn.preprocessing import FunctionTransformer

from aeon.datasets import load_italy_power_demand
from aeon.distances import msm_distance, msm_pairwise_distance

X, y = load_italy_power_demand(split="TRAIN")

# Create a FunctionTransformer to apply the MSM pairwise distance
ft = FunctionTransformer(msm_pairwise_distance)
X2 = ft.transform(X)
print(f"Shape (FunctionTransformer) = {X2.shape}")

# Compute the MSM pairwise distance
X2_bis = msm_pairwise_distance(X)
print(f"Shape (msm_pairwise_distance) = {X2_bis.shape}")

# Check that the three results are the same
d = msm_distance(X[0], X[1])
print(f"These values are the same {d}, {X2[0][1]} and {X2_bis[0][1]}.")


# %%NBQA-CELL-SEPfc780c
from sklearn.pipeline import Pipeline

# Fit a pipeline with the FunctionTransformer (using the msm_pairwise_distance) and the
# SVM classifier
pipe = Pipeline(steps=[("FunctionTransformer", ft), ("SVM", SVC())])
pipe.fit(X, y)

print(
    "Pipeline with SVM and MSM distance works! First five predictions = ",
    pipe.predict(X)[:5],
)
