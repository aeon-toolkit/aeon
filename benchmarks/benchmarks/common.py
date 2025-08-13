import os
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class safe_import:

    def __enter__(self):
        self.error = False
        return self

    def __exit__(self, type_, value, traceback):
        if type_ is not None:
            self.error = True
            suppress = not (
                os.getenv("SCIPY_ALLOW_BENCH_IMPORT_ERRORS", "1").lower()
                in ("0", "false")
                or not issubclass(type_, ImportError)
            )
            return suppress


class Benchmark:
    """
    Base class with sensible options
    """


with safe_import():
    from aeon.testing.data_generation import make_example_3d_numpy


class EstimatorBenchmark(Benchmark, ABC):
    # Base grid (shared across all estimators)
    shapes = [
        (10, 1, 10),
        (100, 1, 100),
        (10, 3, 10),
        (100, 3, 100),
    ]

    # Subclasses will append their own grids to these:
    params = [shapes]
    param_names = ["shape"]

    def setup(self, shape, *est_params):
        # Data
        self.X_train = make_example_3d_numpy(*shape, return_y=False, random_state=1)
        self.X_test = make_example_3d_numpy(*shape, return_y=False, random_state=2)

        # Warm-up tiny run (helps numba/JIT caches etc.)
        tmp_X = make_example_3d_numpy(10, 1, 10, random_state=42, return_y=False)
        tmp_est = self._build_estimator(*est_params)
        for _ in range(3):
            tmp_est.fit(tmp_X)
            tmp_est.predict(tmp_X)

        # Pre-fit once for predict timing
        self.prefit_estimator = self._build_estimator(*est_params)
        self.prefit_estimator.fit(self.X_train)

    def time_fit(self, shape, *est_params):
        est = self._build_estimator(*est_params)  # fresh each run
        est.fit(self.X_train)

    def time_predict(self, shape, *est_params):
        self.prefit_estimator.predict(self.X_test)

    @abstractmethod
    def _build_estimator(self, *est_params) -> BaseEstimator:
        """Return an unfitted estimator configured with the given params."""
        ...
