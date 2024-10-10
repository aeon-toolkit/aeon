"""Test DrCIF regressor."""
from aeon.regression.interval_based import RandomIntervalSpectralEnsembleRegressor

def test_rise():
    """Test with pyfftw enabled."""
    dr = RandomIntervalSpectralEnsembleRegressor(use_pyfftw=True)
    d = dr.get_tag("python_dependencies")
    assert d is "pyfftw"
    paras = RandomIntervalSpectralEnsembleRegressor.get_test_params(parameter_set="contracting")
    assert paras["time_limit_in_minutes"] == 5
    assert paras["contract_max_n_estimators"] == 2

