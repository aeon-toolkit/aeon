"""Tests for HIVE-COTE."""

import numpy as np
import pytest

from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(PR_TESTING, reason="slow test, run overnight only")
def test_hc1_defaults_and_verbosity():
    """Test HC1 default parameters and verbose setting."""
    HIVECOTEV1._DEFAULT_N_TREES = 10
    HIVECOTEV1._DEFAULT_N_SHAPELETS = 10
    HIVECOTEV1._DEFAULT_MAX_ENSEMBLE_SIZE = 5
    HIVECOTEV1._DEFAULT_N_PARA_SAMPLES = 10
    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=10, n_labels=2)
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
def test_hc2_defaults_and_verbosity():
    """Test HC2 default parameters and verbose setting."""
    HIVECOTEV2._DEFAULT_N_TREES = 10
    HIVECOTEV2._DEFAULT_N_SHAPELETS = 10
    HIVECOTEV2._DEFAULT_N_KERNELS = 100
    HIVECOTEV2._DEFAULT_N_ESTIMATORS = 5
    HIVECOTEV2._DEFAULT_N_PARA_SAMPLES = 10
    HIVECOTEV2._DEFAULT_MAX_ENSEMBLE_SIZE = 5
    HIVECOTEV2._DEFAULT_RAND_PARAMS = 5
    X, _ = make_example_3d_numpy(n_cases=20, n_timepoints=20, n_labels=2)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    hc2 = HIVECOTEV2(verbose=True)
    hc2.fit(X, y)
    assert hc2._stc_params == {"n_shapelet_samples": 10}
    assert hc2._drcif_params == {"n_estimators": 10}
    assert hc2._arsenal_params == {"n_kernels": 100, "n_estimators": 5}
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
