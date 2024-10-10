"""Test interval-based regression models."""

from sklearn.ensemble import RandomForestRegressor

from aeon.regression.interval_based import RandomIntervalRegressor
from aeon.testing.data_generation import make_example_3d_numpy


def test_cif():
    """Test with IntervalForestRegressor contracting."""
    cls = RandomIntervalRegressor(
        n_jobs=1,
        n_intervals=5,
        estimator=RandomForestRegressor(n_estimators=10, n_jobs=2),
    )
    X, y = make_example_3d_numpy()
    cls.fit(X, y)
    assert cls._estimator.n_jobs == 1
