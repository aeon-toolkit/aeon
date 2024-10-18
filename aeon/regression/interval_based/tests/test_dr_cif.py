"""Test DrCIF regressor."""

from aeon.regression.interval_based import DrCIFRegressor


def test_dr_cif():
    """Test with pycatch22 enabled."""
    dr = DrCIFRegressor(use_pycatch22=True)
    d = dr.get_tag("python_dependencies")
    assert d[0] == "pycatch22"
    paras = DrCIFRegressor.get_test_params(parameter_set="contracting")
    assert paras["time_limit_in_minutes"] == 5
    assert paras["att_subsample_size"] == 2
