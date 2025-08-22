from sklearn.base import BaseEstimator

from .common import EstimatorBenchmark, safe_import

with safe_import():
    import aeon.clustering as aeon_clust


class KMeansBenchmark(EstimatorBenchmark):
    """This runs kmeans with mean averaging method."""

    inits = ["random", "kmeans++"]

    params = EstimatorBenchmark.params + [inits]
    param_names = EstimatorBenchmark.param_names + ["init"]

    def _build_estimator(self, k, init, distance) -> BaseEstimator:
        return aeon_clust.TimeSeriesKMeans(
            n_clusters=4,
            init=init,
            distance="euclidean",
            averaging_method="mean",
            n_init=1,
            max_iter=20,
            random_state=1,
        )


class KMeansBABenchmark(EstimatorBenchmark):
    """This runs kmeans with ba averaging method."""

    distances = ["dtw", "msm"]
    params = EstimatorBenchmark.params + [
        KMeansBenchmark.inits,
        distances,
    ]
    param_names = EstimatorBenchmark.param_names + ["init", "distance"]

    def _build_estimator(self, init, distance) -> BaseEstimator:
        return aeon_clust.TimeSeriesKMeans(
            n_clusters=4,
            init=init,
            distance=distance,
            averaging_method="ba",
            n_init=1,
            max_iter=10,
            random_state=1,
        )
