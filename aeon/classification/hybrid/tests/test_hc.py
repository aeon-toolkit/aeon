# -*- coding: utf-8 -*-
"""Tests for HC1."""
import pytest

from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from aeon.tests._config import PR_TESTING
from aeon.utils._testing.collection import make_2d_test_data


@pytest.mark.skipif(PR_TESTING, reason="slow test, run overnight only")
def test_hc1():
    """Test HC1."""
    HIVECOTEV1._DEFAULT_N_TREES = 10
    HIVECOTEV1._DEFAULT_N_SHAPELETS = 10
    HIVECOTEV1._DEFAULT_MAX_ENSEMBLE_SIZE = 5
    HIVECOTEV1._DEFAULT_N_PARA_SAMPLES = 10
    X, y = make_2d_test_data(n_cases=20, n_timepoints=10, n_labels=2)
    hc1 = HIVECOTEV1(verbose=True)
    hc1.fit(X, y)
    assert hc1._stc_params == {"n_shapelet_samples": 10}
    assert hc1._tsf_params == {"n_estimators": 10}
    assert hc1._rise_params == {"n_estimators": 10}
    assert hc1._cboss_params == {"n_parameter_samples": 10, "max_ensemble_size": 5}
    HIVECOTEV1._DEFAULT_N_TREES = 500
    HIVECOTEV1._DEFAULT_N_SHAPELETS = 10000
    HIVECOTEV1._DEFAULT_MAX_ENSEMBLE_SIZE = 250
    HIVECOTEV1._DEFAULT_N_PARA_SAMPLES = 50


@pytest.mark.skipif(PR_TESTING, reason="slow test, run overnight only")
def test_hc2():
    """Test HC2."""
    HIVECOTEV2._DEFAULT_N_TREES = 10
    HIVECOTEV2._DEFAULT_N_SHAPELETS = 10
    HIVECOTEV2._DEFAULT_N_KERNELS = 100
    HIVECOTEV2._DEFAULT_N_ESTIMATORS = 5
    HIVECOTEV2._DEFAULT_N_PARA_SAMPLES = 10
    HIVECOTEV2._DEFAULT_MAX_ENSEMBLE_SIZE = 5
    HIVECOTEV2._DEFAULT_RAND_PARAMS = 5
    X, y = make_2d_test_data(n_cases=20, n_timepoints=10, n_labels=2)
    hc2 = HIVECOTEV2(verbose=True)
    hc2.fit(X, y)
    assert hc2._stc_params == {"n_shapelet_samples": 10}
    assert hc2._drcif_params == {"n_estimators": 10}
    assert hc2._arsenal_params == {"num_kernels": 100, "n_estimators": 5}
    assert hc2._tde_params == {
        "n_parameter_samples": 10,
        "max_ensemble_size": 5,
        "randomly_selected_params": 5,
    }
    HIVECOTEV2._DEFAULT_N_TREES = 500
    HIVECOTEV2._DEFAULT_N_SHAPELETS = 10000
    HIVECOTEV2._DEFAULT_N_KERNELS = 2000
    HIVECOTEV2._DEFAULT_N_ESTIMATORS = 25
    HIVECOTEV2._DEFAULT_N_PARA_SAMPLES = 250
    HIVECOTEV2._DEFAULT_MAX_ENSEMBLE_SIZE = 50
    HIVECOTEV2._DEFAULT_RAND_PARAMS = 50
