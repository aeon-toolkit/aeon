from abc import ABC, abstractmethod

from .common import Benchmark, safe_import

with safe_import():
    import aeon.distances as aeon_dists
    from aeon.testing.data_generation import make_example_3d_numpy


class _DistanceBenchmark(Benchmark, ABC):
    # params: (n_cases, n_channels, n_timepoints)
    params = [
        [
            (10, 1, 10),
            (100, 1, 100),
            (100, 1, 500),
            (10, 3, 10),
            (100, 3, 100),
            (100, 3, 500),
        ],
    ]
    param_names = ["shape"]

    def setup(self, shape):
        # Two independent samples so we don't measure duplicates
        self.a = make_example_3d_numpy(*shape, return_y=False, random_state=1)
        self.b = make_example_3d_numpy(*shape, return_y=False, random_state=2)

    def time_indv_dist(self, shape):
        # single-series distance
        self.distance_func(self.a[0], self.b[0])

    def time_pairwise_dist(self, shape):
        # pairwise(X) or pairwise(X, Y)
        self.pairwise_func(self.a)

    def time_one_to_multiple_dist(self, shape):
        self.pairwise_func(self.a[0], self.b)

    def time_multiple_to_multiple_dist(self, shape):
        self.pairwise_func(self.a, self.b)

    @property
    @abstractmethod
    def distance_func(self):
        """Return a callable: dist(Xi, Xj) -> float"""
        raise NotImplementedError

    @property
    @abstractmethod
    def pairwise_func(self):
        """Return a callable: pairwise(X[, Y]) -> ndarray"""
        raise NotImplementedError


class _ElasticDistanceBenchmark(_DistanceBenchmark, ABC):
    def time_alignment_path(self, shape):
        self.alignment_func(self.a[0], self.b[0])

    @property
    @abstractmethod
    def alignment_func(self):
        """Return a callable: alignment_path(Xi, Xj) -> (path, cost) or similar"""
        raise NotImplementedError


class DTW(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.dtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.dtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.dtw_alignment_path


class MSM(_ElasticDistanceBenchmark):

    @property
    def distance_func(self):
        return aeon_dists.msm_distance

    @property
    def pairwise_func(self):
        return aeon_dists.msm_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.msm_alignment_path


class TWE(_ElasticDistanceBenchmark):

    @property
    def distance_func(self):
        return aeon_dists.twe_distance

    @property
    def pairwise_func(self):
        return aeon_dists.twe_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.twe_alignment_path


class Euclidean(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.euclidean_distance

    @property
    def pairwise_func(self):
        return aeon_dists.euclidean_pairwise_distance
