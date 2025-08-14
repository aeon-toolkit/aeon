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
            (10, 1, 1000),
            (50, 1, 100),
            (10, 3, 10),
            (10, 3, 1000),
            (50, 3, 100),
        ],
    ]
    param_names = ["shape"]

    def setup(self, shape):
        self.a = make_example_3d_numpy(*shape, return_y=False, random_state=1)
        self.b = make_example_3d_numpy(*shape, return_y=False, random_state=2)

    def time_dist(self, shape):
        self.distance_func(self.a[0], self.b[0])

    def time_pairwise_dist(self, shape):
        self.pairwise_func(self.a)

    def time_one_to_multiple_dist(self, shape):
        self.pairwise_func(self.a[0], self.b)

    def time_multiple_to_multiple_dist(self, shape):
        self.pairwise_func(self.a, self.b)

    def time_muti_thread_pairwise_dist(self, shape):
        self.pairwise_func(self.a, n_jobs=2)

    def time_muti_thread_dist(self, shape):
        self.pairwise_func(self.a, self.b, n_jobs=4)

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

    def setup(self, shape):
        temp = make_example_3d_numpy(5, 1, 10, random_state=42, return_y=False)
        for _ in range(3):
            self.alignment_func(temp[0], temp[0])

        super().setup(shape)

    def time_alignment_path(self, shape):
        self.alignment_func(self.a[0], self.b[0])

    @property
    @abstractmethod
    def alignment_func(self):
        """Return a callable: alignment_path(Xi, Xj) -> (path, cost) or similar"""
        raise NotImplementedError


class SquaredBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.squared_distance

    @property
    def pairwise_func(self):
        return aeon_dists.squared_pairwise_distance


class ManhattanBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.manhattan_distance

    @property
    def pairwise_func(self):
        return aeon_dists.manhattan_pairwise_distance


class MinkowskiBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.minkowski_distance

    @property
    def pairwise_func(self):
        return aeon_dists.minkowski_pairwise_distance


class DTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.dtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.dtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.dtw_alignment_path


class DTWGIBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.dtw_gi_distance

    @property
    def pairwise_func(self):
        return aeon_dists.dtw_gi_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.dtw_gi_alignment_path


class DDTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.ddtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.ddtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.ddtw_alignment_path


class WDTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.wdtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.wdtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.wdtw_alignment_path


class WDDTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.wddtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.wddtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.wddtw_alignment_path


class LCSSBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.lcss_distance

    @property
    def pairwise_func(self):
        return aeon_dists.lcss_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.lcss_alignment_path


class ERPBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.erp_distance

    @property
    def pairwise_func(self):
        return aeon_dists.erp_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.erp_alignment_path


class EDRBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.edr_distance

    @property
    def pairwise_func(self):
        return aeon_dists.edr_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.edr_alignment_path


class TWEBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.twe_distance

    @property
    def pairwise_func(self):
        return aeon_dists.twe_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.twe_alignment_path


class MSMBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.msm_distance

    @property
    def pairwise_func(self):
        return aeon_dists.msm_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.msm_alignment_path


class ADTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.adtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.adtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.adtw_alignment_path


class ShapeDTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.shape_dtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.shape_dtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.shape_dtw_alignment_path


class SoftDTWBenchmark(_ElasticDistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.soft_dtw_distance

    @property
    def pairwise_func(self):
        return aeon_dists.soft_dtw_pairwise_distance

    @property
    def alignment_func(self):
        return aeon_dists.soft_dtw_alignment_path


class SBDBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.sbd_distance

    @property
    def pairwise_func(self):
        return aeon_dists.sbd_pairwise_distance


class ShiftScaleBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.shift_scale_invariant_distance

    @property
    def pairwise_func(self):
        return aeon_dists.shift_scale_invariant_pairwise_distance


class DFTSFABenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.mindist_dft_sfa_distance

    @property
    def pairwise_func(self):
        return aeon_dists.mindist_dft_sfa_pairwise_distance


class PAASAXBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.mindist_paa_sax_distance

    @property
    def pairwise_func(self):
        return aeon_dists.mindist_paa_sax_pairwise_distance


class SAXBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.mindist_sax_distance

    @property
    def pairwise_func(self):
        return aeon_dists.mindist_sax_pairwise_distance


class SFABenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.mindist_sfa_distance

    @property
    def pairwise_func(self):
        return aeon_dists.mindist_sfa_pairwise_distance


class MPDistBenchmark(_DistanceBenchmark):
    @property
    def distance_func(self):
        return aeon_dists.mp_distance

    @property
    def pairwise_func(self):
        return aeon_dists.mp_pairwise_distance
