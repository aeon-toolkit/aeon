# %%NBQA-CELL-SEPfc780c
from aeon.datasets import (
    load_anomaly_detection,
    load_classification,
    load_forecasting,
    load_regression,
)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.tsc_datasets import multivariate, univariate

# This file also contains sub lists by type, e.g. unequal length
print("Univariate length = ", len(univariate))
print("Multivariate length = ", len(multivariate))


# %%NBQA-CELL-SEPfc780c
X, y, meta = load_classification("Chinatown", return_metadata=True)
print("Shape of X = ", X.shape)
print("First case = ", X[0][0], " has label = ", y[0])
print("\nMeta data = ", meta)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.dataset_collections import get_available_tser_datasets

get_available_tser_datasets()


# %%NBQA-CELL-SEPfc780c
X, y, meta = load_regression("FloodModeling1", return_metadata=True)
print("Shape of X = ", X.shape, " meta data = ", meta)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.dataset_collections import get_available_tsf_datasets

get_available_tsf_datasets()


# %%NBQA-CELL-SEPfc780c
X, metadata = load_forecasting("m4_yearly_dataset", return_metadata=True)
print(X.shape)
print(metadata)
data = X.head()
print(data)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.tsad_datasets import multivariate, univariate

# This file also contains sub lists by learning type, e.g. semi-supervised, ...
print("Univariate length = ", len(univariate()))
print("Multivariate length = ", len(multivariate()))


# %%NBQA-CELL-SEPfc780c
name = ("Genesis", "genesis-anomalies")
X, y, meta = load_anomaly_detection(name, return_metadata=True)
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)
print("\nMeta data = ", meta)
