"""Test DrCIF regressor."""

from aeon.regression.interval_based import CanonicalIntervalForestRegressor


def test_cif():
    """Test with catch22 enabled."""
    dr = CanonicalIntervalForestRegressor(use_pycatch22=True)
    d = dr.get_tag("python_dependencies")
    assert d == "pycatch22"
    paras = CanonicalIntervalForestRegressor._get_test_params(
        parameter_set="contracting"
    )
    assert paras["time_limit_in_minutes"] == 5
    assert paras["att_subsample_size"] == 2
