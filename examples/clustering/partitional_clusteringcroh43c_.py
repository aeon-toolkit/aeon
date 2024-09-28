# %%NBQA-CELL-SEPfc780c
# Imports and load data
from aeon.clustering import TimeSeriesKMeans, TimeSeriesKMedoids
from aeon.datasets import load_unit_test
from aeon.visualisation import plot_cluster_algorithm

X_train, y_train = load_unit_test(split="train")
(
    X_test,
    y_test,
) = load_unit_test(split="test")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# %%NBQA-CELL-SEPfc780c
temp = TimeSeriesKMeans(
    init_algorithm="kmeans++",  # initialisation technique: random, or kmeans++
)
print(temp.init_algorithm)


# %%NBQA-CELL-SEPfc780c
TimeSeriesKMeans(
    distance="dtw",  # DTW distance with 25% warping allowed
    distance_params={"window": 0.25},
)


# %%NBQA-CELL-SEPfc780c
temp = TimeSeriesKMeans(distance="euclidean", averaging_method="mean")
print(temp.get_params())
TimeSeriesKMeans(
    distance="msm", averaging_method="ba", average_params={"max_iters": 10}
)


# %%NBQA-CELL-SEPfc780c
k_means = TimeSeriesKMeans(
    n_clusters=2,  # Number of desired centers
    init="random",  # initialisation technique: random, first or kmeans++
    max_iter=10,  # Maximum number of iterations for refinement on training set
    distance="dtw",  # Distance metric to use
    averaging_method="mean",  # Averaging technique to use
    random_state=1,  # Makes deterministic
)

k_means.fit(X_train)
plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)


# %%NBQA-CELL-SEPfc780c
s1 = k_means.score(X_test, y_test)
s1


# %%NBQA-CELL-SEPfc780c
# Best configuration for k-means
k_means = TimeSeriesKMeans(
    n_clusters=2,  # Number of desired centers
    init="random",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    distance="msm",  # Distance metric to use
    averaging_method="ba",  # Averaging technique to use
    random_state=1,
    average_params={
        "distance": "msm",
    },
)

k_means.fit(X_train)
plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)


# %%NBQA-CELL-SEPfc780c
s2 = k_means.score(X_test, y_test)
s2


# %%NBQA-CELL-SEPfc780c
temp2 = TimeSeriesKMedoids(
    distance="msm",  # MSM distance with c parameter set to 0.2 and 90% window.
    distance_params={"c": 2.0, "window": 0.9, "independent": True},
)
print(temp.distance, ", ", temp.distance_params)
print(temp2.distance, ", ", temp2.distance_params)


# %%NBQA-CELL-SEPfc780c
k_medoids = TimeSeriesKMedoids(
    n_clusters=2,  # Number of desired centers
    init="random",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    verbose=False,  # Verbose
    distance="dtw",  # Distance to use
    random_state=1,
)

k_medoids.fit(X_train)
s3 = k_medoids.score(X_test, y_test)
plot_cluster_algorithm(k_medoids, X_test, k_medoids.n_clusters)


# %%NBQA-CELL-SEPfc780c
k_medoids = TimeSeriesKMedoids(
    n_clusters=2,  # Number of desired centers
    init="random",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    distance="msm",  # Distance to use
    random_state=1,
)

k_medoids.fit(X_train)
s4 = k_medoids.score(X_test, y_test)
plot_cluster_algorithm(k_medoids, X_test, k_medoids.n_clusters)


# %%NBQA-CELL-SEPfc780c
print(f" PAM DTW score {s3} PAM MSM score {s4}")


# %%NBQA-CELL-SEPfc780c
k_medoids = TimeSeriesKMedoids(
    n_clusters=2,  # Number of desired centers
    init="random",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    distance="msm",  # Distance to use
    random_state=1,
    method="alternate",
)

k_medoids.fit(X_train)
s5 = k_medoids.score(X_test, y_test)
plot_cluster_algorithm(k_medoids, X_test, k_medoids.n_clusters)
print("Alternate MSM score = ", s5)


# %%NBQA-CELL-SEPfc780c
from aeon.clustering import TimeSeriesCLARA

clara = TimeSeriesCLARA(
    n_clusters=2,  # Number of desired centers
    max_iter=10,  # Maximum number of iterations for refinement on training set
    distance="msm",  # Distance to use
    random_state=1,
)
clara.fit(X_train)
s6 = k_medoids.score(X_test, y_test)
plot_cluster_algorithm(clara, X_test, clara.n_clusters)


# %%NBQA-CELL-SEPfc780c
from aeon.clustering import TimeSeriesCLARANS

clara = TimeSeriesCLARANS(
    n_clusters=2,  # Number of desired centers
    distance="msm",  # Distance to use
    random_state=1,
)
clara.fit(X_train)
s7 = k_medoids.score(X_test, y_test)
plot_cluster_algorithm(clara, X_test, clara.n_clusters)


# %%NBQA-CELL-SEPfc780c
print(f" Clara score {s6} Clarans score = {s7}")


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.visualisation import plot_critical_difference

# 1. MSM is the most effective distance function
clusterers = [
    "kmeans-ed",
    "kmeans-dtw",
    "kmeans-msm",
    "kmeans-twe",
    "kmedoids-ed",
    "kmedoids-dtw",
    "kmedoids-msm",
    "kmedoids-twe",
]
accuracy, data_names = get_estimator_results_as_array(
    task="clustering", estimators=clusterers, measure="clacc"
)
print(f" Returned results in shape {accuracy.shape}")
plt = plot_critical_difference(accuracy, clusterers)
