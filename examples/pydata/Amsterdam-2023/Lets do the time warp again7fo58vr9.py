# %%NBQA-CELL-SEPfc780c
import numpy as np

# These can be 1d or 2d arrays and in addition they do not have to be equal length
x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])

# Calling a distance directly
from aeon.distances import dtw_distance

dtw_distance(x, y)

# Calling a distance using utility function
from aeon.distances import distance

# Any value in the table above is a valid metric string
distance(x, y, metric="dtw")


# %%NBQA-CELL-SEPfc780c
# Calling a distance directly
from aeon.distances import msm_distance

msm_distance(x, y, window=0.2)

# Calling a distance using utility function
distance(x, y, metric="msm", window=0.2)


# %%NBQA-CELL-SEPfc780c
X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

# Calling a distance directly
from aeon.distances import twe_pairwise_distance

twe_pairwise_distance(X)

# Using utility function
from aeon.distances import pairwise_distance

pairwise_distance(X, metric="twe")


# %%NBQA-CELL-SEPfc780c
y = np.array([[16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])

# Calling a distance directly
from aeon.distances import erp_pairwise_distance

erp_pairwise_distance(X, y)

# Using utility function
pairwise_distance(X, y, metric="erp")


# %%NBQA-CELL-SEPfc780c
# Calling a distance directly
from aeon.distances import wdtw_pairwise_distance

wdtw_pairwise_distance(X, y, window=0.2)

# Using utility function
pairwise_distance(X, y, metric="wdtw", window=0.2)


# %%NBQA-CELL-SEPfc780c
x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])
# Calling a alignment path directly
from aeon.distances import msm_alignment_path

msm_alignment_path(x, y)

# Using utility function
from aeon.distances import alignment_path

alignment_path(x, y, metric="msm", window=0.2)


# %%NBQA-CELL-SEPfc780c
# We can visualise the alignment path using the plot_alignment_path function


def setup_time_axes(data: np.ndarray):
    fig, ax = plt.subplots(1, 1, figsize=(20, 4), dpi=300)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.grid(color="white", linewidth=0.1, alpha=0.3, zorder=0)
    ax.axis("off")

    return fig, ax


def plot_alignment(x, y, path):
    time = np.arange(len(x))
    fig, ax = setup_time_axes(np.vstack([x, y]))
    # Plot the first time series
    ax.plot(time, x, marker="o", linestyle="-")

    # Plot the second time series
    ax.plot(time, y, marker="x", linestyle="-")

    for i, j in path:
        ax.plot([time[i], time[j]], [x[i], y[j]], "--", linewidth=0.5, color="gray")

    return plt


# %%NBQA-CELL-SEPfc780c
import matplotlib.pyplot as plt

from aeon.datasets import load_gunpoint as load_data

X_train, y_train = load_data(split="TRAIN", return_type="numpy2D")
X_test, y_test = load_data(split="TEST", return_type="numpy2D")
x = X_train[0]
y = X_train[22]
msm_alignment_path = alignment_path(x, y, metric="msm")
curr_plot = plot_alignment(x, y, msm_alignment_path[0])
curr_plot.show()


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_gunpoint as load_data

X_train, y_train = load_data(split="TRAIN", return_type="numpy2D")
X_test, y_test = load_data(split="TEST", return_type="numpy2D")


# %%NBQA-CELL-SEPfc780c
from sklearn.cluster import AgglomerativeClustering

from aeon.distances import pairwise_distance, twe_pairwise_distance

model_precomputed = AgglomerativeClustering(metric="precomputed", linkage="complete")
model_distance = AgglomerativeClustering(
    metric=twe_pairwise_distance, linkage="complete"
)

# Precompute pairwise twe distances
train_pw_distance = pairwise_distance(X_train, metric="twe")

# Fit model using precomputed
model_precomputed.fit(train_pw_distance)
# Fit model using distance function
model_distance.fit(X_train)
#
# Score models on training data
print("DBSCAN with twe distance labels: ", model_distance.labels_)
print("DBSCAN with precomputed labels: ", model_precomputed.labels_)


# %%NBQA-CELL-SEPfc780c
from sklearn.svm import SVC

from aeon.distances import msm_pairwise_distance, pairwise_distance

model_precomputed = SVC(kernel="precomputed")
model_distance = SVC(kernel=msm_pairwise_distance)

# Precompute pairwise twe distances
train_pw_distance = pairwise_distance(X_train, metric="msm")
test_pw_distance = pairwise_distance(X_test, X_train, metric="msm")

# Fit model using precomputed
model_precomputed.fit(train_pw_distance, y_train)
# Fit model using distance function
model_distance.fit(X_train, y_train)

# Score models on training data
print("SVM with twe distance score: ", model_distance.score(X_test, y_test))
print("SVM with precomputed score: ", model_precomputed.score(test_pw_distance, y_test))
