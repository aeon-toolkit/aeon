"""Test Inter regressor."""

from aeon.regression.interval_based import IntervalForestRegressor


def test_cif():
    """Test with IntervalForestRegressor contracting."""
    paras = IntervalForestRegressor._get_test_params(parameter_set="contracting")
    assert paras["time_limit_in_minutes"] == 5
    assert paras["n_intervals"] == 2
