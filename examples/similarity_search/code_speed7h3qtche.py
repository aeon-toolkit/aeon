# %%NBQA-CELL-SEPfc780c
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aeon.distances import squared_distance
from aeon.similarity_search.distance_profiles.naive_distance_profile import (
    naive_distance_profile,
    normalized_naive_distance_profile,
)
from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    normalized_squared_distance_profile,
    squared_distance_profile,
)
from aeon.utils.numba.general import sliding_mean_std_one_series

ggplot_styles = {
    "axes.edgecolor": "white",
    "axes.facecolor": "EBEBEB",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "grid.color": "white",
    "grid.linewidth": "1.2",
    "xtick.color": "555555",
    "xtick.major.bottom": True,
    "xtick.minor.bottom": False,
    "ytick.color": "555555",
    "ytick.major.left": True,
    "ytick.minor.left": False,
}

plt.rcParams.update(ggplot_styles)


# %%NBQA-CELL-SEPfc780c
def rolling_window_stride_trick(X, window):
    """
    Use strides to generate rolling/sliding windows for a numpy array.
    Parameters
    ----------
    X : numpy.ndarray
        numpy array
    window : int
        Size of the rolling window
    Returns
    -------
    output : numpy.ndarray
        This will be a new view of the original input array.
    """
    a = np.asarray(X)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_means_stds(X, query_length):
    windows = rolling_window_stride_trick(X, query_length)
    return windows.mean(axis=-1), windows.std(axis=-1)


rng = np.random.default_rng()
size = 100
query_length = 10

# Create a random series with 1 feature and 'size' timesteps
X = rng.random((1, size))
means, stds = get_means_stds(X, query_length)
print(means.shape)


# %%NBQA-CELL-SEPfc780c
sizes = [500, 1000, 5000, 10000, 25000, 50000]
query_lengths = [50, 100, 250, 500]
times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)
# A first run for numba compilations if needed
sliding_mean_std_one_series(rng.random((1, 50)), 10, 1)
for size in sizes:
    for query_length in query_lengths:
        X = rng.random((1, size))
        _times = hash("b03aee26")
        times.loc[(size, query_length), "full computation"] = _times.average
        _times = hash("a910a3eb")
        times.loc[(size, query_length), "sliding_computation"] = _times.average


# %%NBQA-CELL-SEPfc780c
fig, ax = plt.subplots(ncols=len(query_lengths), figsize=(20, 5), dpi=200, sharey=True)
for j, (i, grp) in enumerate(times.groupby("query_length")):
    grp.droplevel(1).plot(label=i, ax=ax[j])
    ax[j].set_title(f"query length {i}")
ax[0].set_ylabel("time in seconds")
plt.show()


# %%NBQA-CELL-SEPfc780c
rng = np.random.default_rng()

sizes = [500, 1000, 5000, 10000, 20000, 30000, 50000]
query_lengths = [0.01, 0.05, 0.1, 0.2]
times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)

for size in sizes:
    for _query_length in query_lengths:
        query_length = int(_query_length * size)
        X = rng.random((1, 1, size))
        q = rng.random((1, query_length))
        mask = np.ones((1, size - query_length + 1), dtype=bool)
        # Used for numba compilation before timings
        naive_distance_profile(X, q, mask, squared_distance)
        _times = hash("6848ae1f")
        times.loc[(size, _query_length), "Naive Euclidean distance"] = _times.average
        # Used for numba compilation before timings
        squared_distance_profile(X, q, mask)
        _times = hash("ae467853")
        times.loc[(size, _query_length), "Euclidean distance as dot product"] = (
            _times.average
        )


# %%NBQA-CELL-SEPfc780c
fig, ax = plt.subplots(ncols=len(query_lengths), figsize=(20, 5), dpi=200)
for j, (i, grp) in enumerate(times.groupby("query_length")):
    grp.droplevel(1).plot(label=i, ax=ax[j])
    ax[j].set_title(f"query length {i}")
ax[0].set_ylabel("time in seconds")
plt.show()


# %%NBQA-CELL-SEPfc780c
sizes = [500, 1000, 5000, 10000, 20000, 30000, 50000]
query_lengths = [0.01, 0.05, 0.1, 0.2]
times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)

for size in sizes:
    for _query_length in query_lengths:
        query_length = int(_query_length * size)
        X = rng.random((1, 1, size))
        q = rng.random((1, query_length))
        n_cases, n_channels = X.shape[0], X.shape[1]
        search_space_size = size - query_length + 1
        X_means = np.zeros((n_cases, n_channels, search_space_size))
        X_stds = np.zeros((n_cases, n_channels, search_space_size))
        mask = np.ones((n_channels, search_space_size), dtype=bool)
        for i in range(X.shape[0]):
            _mean, _std = sliding_mean_std_one_series(X[i], query_length, 1)
            X_stds[i] = _std
            X_means[i] = _mean
        q_means, q_stds = sliding_mean_std_one_series(q, query_length, 1)
        q_means = q_means[:, 0]
        q_stds = q_stds[:, 0]
        # Used for numba compilation before timings
        normalized_naive_distance_profile(
            X, q, mask, X_means, X_stds, q_means, q_stds, squared_distance
        )
        _times = hash("55982fec")
        times.loc[(size, _query_length), "Naive Normalized Euclidean distance"] = (
            _times.average
        )
        # Used for numba compilation before timings
        normalized_squared_distance_profile(
            X, q, mask, X_means, X_stds, q_means, q_stds
        )
        _times = hash("eabc1b97")
        times.loc[(size, _query_length), "Normalized Euclidean as dot product"] = (
            _times.average
        )


# %%NBQA-CELL-SEPfc780c
fig, ax = plt.subplots(ncols=len(query_lengths), figsize=(20, 5), dpi=200)
for j, (i, grp) in enumerate(times.groupby("query_length")):
    grp.droplevel(1).plot(label=i, ax=ax[j])
    ax[j].set_title(f"query length {i}")
ax[0].set_ylabel("time in seconds")
plt.show()


# %%NBQA-CELL-SEPfc780c
from aeon.similarity_search._commons import get_ith_products
from aeon.similarity_search.matrix_profiles.stomp import _update_dot_products_one_series


def compute_all_products(X, T, L):
    for i in range(T.shape[1] - L + 1):
        prods = get_ith_products(X, T, L, i)
    return prods


def update_products(X, T, L):
    prods = get_ith_products(X, T, L, 0)
    for i in range(T.shape[1] - L + 1):
        prods = _update_dot_products_one_series(X, T, prods, L, i)
    return prods


sizes = [500, 1000, 5000, 10000]
query_lengths = [0.01, 0.05, 0.1]
times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)

times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)

for size in sizes:
    for _query_length in query_lengths:
        query_length = int(_query_length * size)
        X = rng.random((1, size))
        T = rng.random((1, size))
        search_space_size = size - query_length + 1
        mask = np.ones((1, search_space_size), dtype=bool)
        # Used for numba compilation before timings
        compute_all_products(X, T, query_length)
        _times = hash("71efd768")
        times.loc[(size, _query_length), "compute_all_products"] = _times.average
        # Used for numba compilation before timings
        update_products(X, T, query_length)
        _times = hash("fc54337f")
        times.loc[(size, _query_length), "update_products"] = _times.average


# %%NBQA-CELL-SEPfc780c
fig, ax = plt.subplots(ncols=len(query_lengths), figsize=(20, 5), dpi=200)
for j, (i, grp) in enumerate(times.groupby("query_length")):
    grp.droplevel(1).plot(label=i, ax=ax[j])
    ax[j].set_title(f"query length {i}")
ax[0].set_ylabel("time in seconds")
plt.show()


# %%NBQA-CELL-SEPfc780c
from aeon.similarity_search.matrix_profiles import (
    naive_matrix_profile,
    stomp_squared_matrix_profile,
)

sizes = [500, 750, 1000, 3000]
query_lengths = [0.01, 0.05, 0.1]
times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)

times = pd.DataFrame(
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["size", "query_length"])
)

for size in sizes:
    for _query_length in query_lengths:
        query_length = int(_query_length * size)
        X = rng.random((1, 1, size))
        T = rng.random((1, size))
        search_space_size = size - query_length + 1
        mask = np.ones((1, search_space_size), dtype=bool)
        # Used for numba compilation before timings
        naive_matrix_profile(X, T, query_length, distance=squared_distance)
        _times = hash("7e879dd8")
        times.loc[(size, _query_length), "Naive"] = _times.average
        # Used for numba compilation before timings
        stomp_squared_matrix_profile(X, T, query_length, mask)
        _times = hash("95db41b7")
        times.loc[(size, _query_length), "Stomp"] = _times.average


# %%NBQA-CELL-SEPfc780c
fig, ax = plt.subplots(ncols=len(query_lengths), figsize=(20, 5), dpi=200)
for j, (i, grp) in enumerate(times.groupby("query_length")):
    grp.droplevel(1).plot(label=i, ax=ax[j])
    ax[j].set_title(f"query length {i}")
ax[0].set_ylabel("time in seconds")
plt.show()
