# %%NBQA-CELL-SEPfc780c
def plot_best_matches(top_k_search, best_matches):
    fig, ax = plt.subplots(figsize=(20, 5), ncols=3)
    for i_k, (id_sample, id_timestamp) in enumerate(best_matches):
        # plot the sample of the best match
        ax[i_k].plot(top_k_search.X_[id_sample, 0], linewidth=2)
        # plot the location of the best match on it
        ax[i_k].plot(
            range(id_timestamp, id_timestamp + q.shape[1]),
            top_k_search.X_[id_sample, 0, id_timestamp : id_timestamp + q.shape[1]],
            linewidth=7,
            alpha=0.5,
            color="green",
            label="best match location",
        )
        # plot the query on the location of the best match
        ax[i_k].plot(
            range(id_timestamp, id_timestamp + q.shape[1]),
            q[0],
            linewidth=5,
            alpha=0.5,
            color="red",
            label="query",
        )
        ax[i_k].set_title(f"best match {i_k}")
        ax[i_k].legend()
    plt.show()


def plot_matrix_profile(X, mp, i_k):
    fig, ax = plt.subplots(figsize=(20, 10), nrows=2)
    ax[0].set_title("series X used to build the matrix profile")
    ax[0].plot(X[0])  # plot first channel only
    # This is necessary as mp is a list of arrays due to unequal length support
    # as it can have different number of matches for each step when
    # using threshold-based search.
    ax[1].plot([mp[i][i_k] for i in range(len(mp))])
    ax[1].set_title(f"Top {i_k+1} matrix profile of X")
    ax[1].set_ylabel(f"Dist to top {i_k+1} match")
    ax[1].set_xlabel("Starting index of the query in X")
    plt.show()


# %%NBQA-CELL-SEPfc780c
import numpy as np
from matplotlib import pyplot as plt

from aeon.datasets import load_classification

# Load GunPoint dataset
X, y = load_classification("GunPoint")

classes = np.unique(y)

fig, ax = plt.subplots(figsize=(20, 5), ncols=len(classes))
for i_class, _class in enumerate(classes):
    for i_x in np.where(y == _class)[0][0:2]:
        ax[i_class].plot(X[i_x, 0], label=f"sample {i_x}")
    ax[i_class].legend()
    ax[i_class].set_title(f"class {_class}")
plt.suptitle("Example samples for the GunPoint dataset")
plt.show()


# %%NBQA-CELL-SEPfc780c
# We will use the fourth sample an testing data
X_test = X[3]
mask = np.ones(X.shape[0], dtype=bool)
mask[3] = False
# Use this mask to exluce the sample from which we will extract the query
X_train = X[mask]

q = X_test[:, 20:55]
plt.plot(q[0])
plt.show()


# %%NBQA-CELL-SEPfc780c
from aeon.similarity_search import QuerySearch

# Here, the distance function (distance and normalize arguments)
top_k_search = QuerySearch(k=3, distance="euclidean")
# Call fit to store X_train as the database to search in
top_k_search.fit(X_train)
distances_to_matches, best_matches = top_k_search.predict(q)
for i in range(len(best_matches)):
    print(f"match {i} : {best_matches[i]} with distance {distances_to_matches[i]} to q")


# %%NBQA-CELL-SEPfc780c
plot_best_matches(top_k_search, best_matches)


# %%NBQA-CELL-SEPfc780c
# Here, the distance function (distance and normalize arguments)
top_k_search = QuerySearch(k=np.inf, threshold=0.25, distance="euclidean")
top_k_search.fit(X_train)
distances_to_matches, best_matches = top_k_search.predict(q)
for i in range(len(best_matches)):
    print(f"match {i} : {best_matches[i]} with distance {distances_to_matches[i]} to q")


# %%NBQA-CELL-SEPfc780c
# Here, the distance function (distance and normalize arguments)
top_k_search = QuerySearch(k=3, threshold=0.25, distance="euclidean")
top_k_search.fit(X_train)
distances_to_matches, best_matches = top_k_search.predict(q)
for i in range(len(best_matches)):
    print(f"match {i} : {best_matches[i]} with distance {distances_to_matches[i]} to q")


# %%NBQA-CELL-SEPfc780c
# Here, the distance function (distance and normalize arguments)
top_k_search = QuerySearch(k=3, inverse_distance=True, distance="euclidean")
top_k_search.fit(X_train)
distances_to_matches, best_matches = top_k_search.predict(q)
plot_best_matches(top_k_search, best_matches)


# %%NBQA-CELL-SEPfc780c
from aeon.distances import get_distance_function_names

get_distance_function_names()


# %%NBQA-CELL-SEPfc780c
top_k_search = QuerySearch(k=3, distance="dtw", distance_args={"w": 0.2})

q = X[3, :, 20:55]
mask = np.ones(X.shape[0], dtype=bool)
mask[3] = False
# Use this mask to exluce the sample from which we extracted the query
X_train = X[mask]
# Call fit to store X_train as the database to search in
top_k_search.fit(X_train)
distances_to_matches, best_matches = top_k_search.predict(q)
print(best_matches)


# %%NBQA-CELL-SEPfc780c
plot_best_matches(top_k_search, best_matches)


# %%NBQA-CELL-SEPfc780c
def my_dist(x, y):
    return np.sum(np.abs(x - y))


top_k_search = QuerySearch(k=3, distance=my_dist)

q = X[3, :, 20:55]
mask = np.ones(X.shape[0], dtype=bool)
mask[3] = False
# Use this mask to exluce the sample from which we extracted the query
X_train = X[mask]
# Call fit to store X_train as the database to search in
top_k_search.fit(X_train)
distances_to_matches, best_matches = top_k_search.predict(q)
print(best_matches)


# %%NBQA-CELL-SEPfc780c
plot_best_matches(top_k_search, best_matches)


# %%NBQA-CELL-SEPfc780c
QuerySearch.get_speedup_function_names()


# %%NBQA-CELL-SEPfc780c
top_k_search = QuerySearch(distance="euclidean", normalize=True, speed_up="Mueen")


# %%NBQA-CELL-SEPfc780c
from aeon.similarity_search import SeriesSearch

query_length = 35
estimator = SeriesSearch(distance="euclidean").fit(X_train)  # X_test is a 3D array
mp, ip = estimator.predict(X_test, query_length)  # X_test is a 2D array
plot_matrix_profile(X_test, mp, 0)
print(f"Index of the 20-th query best matches : {ip[20]}")
