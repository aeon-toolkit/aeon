# %%NBQA-CELL-SEPfc780c
import warnings

import numpy as np

from aeon.distances import dtw_distance, euclidean_distance

warnings.filterwarnings("ignore")
a = np.array([1, 2, 3, 4, 5, 6])  # Univariate as 1D
b = np.array([2, 3, 4, 5, 6, 7])
d1 = euclidean_distance(a, b)
d2 = dtw_distance(a, b)
print(f" ED 1 = {d1} DTW 1 = {d2}")
x = np.array([[1, 2, 3, 4, 5, 6]])  # Univariate as 2D
y = np.array([[2, 3, 4, 5, 6, 7]])
d1 = euclidean_distance(x, y)
d2 = dtw_distance(x, y)
print(f" ED 2 = {d1} DTW 2 = {d2}")
x = np.array([[1, 2, 3, 4, 5, 6], [3, 4, 3, 4, 3, 4]])  # Multivariate, 2 channels
y = np.array([[2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2]])
d1 = euclidean_distance(x, y)
d2 = dtw_distance(x, y)
print(f" ED 3 = {d1} DTW 3 = {d2}")


# %%NBQA-CELL-SEPfc780c
import matplotlib.pyplot as plt
import numpy as np

from aeon.datasets import load_gunpoint

X, y = load_gunpoint()
print(X.shape)
first = X[1][0]
second = X[2][0]
third = X[4][0]
plt.plot(first, label="First series")
plt.plot(second, label="Second series")
plt.plot(third, label="Third series")
plt.legend()
print(f" class values {y[1]}, {y[2]}, {y[4]}")


# %%NBQA-CELL-SEPfc780c
from aeon.distances import distance, euclidean_distance

d1 = euclidean_distance(first, second)
d2 = euclidean_distance(first, third)
d3 = distance(second, third, metric="euclidean")
print(d1, ",", d2, ",", d3)


# %%NBQA-CELL-SEPfc780c
from aeon.distances import dtw_distance

d1 = dtw_distance(first, second)
d2 = dtw_distance(first, third)
d3 = dtw_distance(second, third)
print(d1, ",", d2, ",", d3)


# %%NBQA-CELL-SEPfc780c
from aeon.distances import create_bounding_matrix

first_ts_size = 10
second_ts_size = 10

create_bounding_matrix(first_ts_size, second_ts_size)


# %%NBQA-CELL-SEPfc780c
create_bounding_matrix(first_ts_size, second_ts_size, window=0.2)


# %%NBQA-CELL-SEPfc780c
a = np.array([1, 3, 4, 5, 7, 10, 3, 2, 1])
b = np.array([1, 4, 5, 7, 10, 3, 2, 1, 1])
c = np.array([1, 2, 5, 5, 5, 5, 5, 2, 1])
plt.plot(a, label="Series a")
plt.plot(b, label="Series b")
plt.plot(c, label="Series c")
plt.legend()
d1 = euclidean_distance(a, b)
d2 = euclidean_distance(a, c)
print("Euclidean distance a to b =", d1)
print("Euclidean distance a to c =", d2)
d1 = dtw_distance(a, b, window=0.0)
d2 = dtw_distance(a, b, window=1.0)
d3 = dtw_distance(a, b, window=0.2)
d4 = euclidean_distance(a, b) ** 2
print("Zero window DTW distance (Squared Euclidean) from a to b =", d1)
print("Squared Euclidean = ", d4)
print("DTW distance (full window) a to b =", d2)
print("DTW distance (20% warping window) a to b =", d3)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_arrow_head
from aeon.distances import msm_pairwise_distance

X_train, _ = load_arrow_head(split="train")
X_test, _ = load_arrow_head(split="test")
X1 = X_train[:5]
X2 = X_test[:6]
train_dist_matrix = msm_pairwise_distance(X1)
test_dist_matrix = msm_pairwise_distance(X1, X2)
print(
    f"Single X dist pairwise is square and symmetrical shape "
    f"= {train_dist_matrix.shape}\n{train_dist_matrix}"
)
print(
    f"Two X dist pairwise is all dists from X1 (row) to X2 (column), so shape "
    f"shape = {test_dist_matrix.shape}\n{test_dist_matrix}"
)


# %%NBQA-CELL-SEPfc780c
from aeon.distances import alignment_path, dtw_alignment_path

x = np.array([[1, 2, 3, 4, 5, 6]])  # Univariate as 2D
y = np.array([[2, 3, 4, 5, 6, 7]])
p, d = dtw_alignment_path(x, y)
print("path =", p, " distance = ", d)
p, d = alignment_path(x, y, metric="dtw")
print("path =", p, " distance = ", d)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_basic_motions

motions, _ = load_basic_motions()
plt.plot(motions[0][0], label="First channel")
plt.plot(motions[0][1], label="Second channel")
plt.plot(motions[0][2], label="Third channel")
plt.legend()


# %%NBQA-CELL-SEPfc780c
d1 = dtw_distance(motions[0], motions[1])
print("Dependent DTW distance (multivariate) motions[0] to motions[1] =", d1)
s = 0
for k in range(motions[0].shape[0]):
    s += dtw_distance(motions[0][k], motions[1][k])
print("Independent DTW distance (multivariate) motions[0] to motions[1] =", s)


# %%NBQA-CELL-SEPfc780c
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7, 8])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[4, 5, 6, 7, 8], [1, 2, 3, 4, 5]])

d1 = dtw_distance(a, b)
d2 = dtw_distance(c, d)
print("Unequal length DTW distance (univariate) =", d1)
print("Unequal length DTW distance (multivariate) =", d2)
