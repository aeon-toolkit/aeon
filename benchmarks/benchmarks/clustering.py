from sklearn.base import BaseEstimator

from .common import EstimatorBenchmark, safe_import

with safe_import():
    import aeon.clustering as aeon_clust
    from aeon.distances._distance import DISTANCES_DICT, ELASTIC_DISTANCES


class KMeansBenchmark(EstimatorBenchmark):
    """This runs kmeans with mean averaging method."""

    ks = [2, 4, 8]
    inits = ["random", "kmeans++"]
    distances = tuple(DISTANCES_DICT.keys())  # all supported distances
    distances = ["euclidean"]

    # base grid
    params = EstimatorBenchmark.params + [ks, inits, distances]
    param_names = EstimatorBenchmark.param_names + ["k", "init", "distance"]

    def _build_estimator(self, k, init, distance) -> BaseEstimator:
        return aeon_clust.TimeSeriesKMeans(
            n_clusters=k,
            init=init,
            distance=distance,
            averaging_method="mean",
            n_init=1,
            max_iter=10,
            random_state=1,
        )


class KMeansBABenchmark(KMeansBenchmark):
    """This runs kmeans with ba averaging method."""

    distances = tuple(ELASTIC_DISTANCES)
    params = EstimatorBenchmark.params + [
        KMeansBenchmark.ks,
        KMeansBenchmark.inits,
        distances,
    ]

    def _build_estimator(self, k, init, distance) -> BaseEstimator:
        # distance will already be elastic-only due to the overridden grid
        return aeon_clust.TimeSeriesKMeans(
            n_clusters=k,
            init=init,
            distance=distance,
            averaging_method="ba",
            n_init=1,
            max_iter=10,
            random_state=1,
        )
