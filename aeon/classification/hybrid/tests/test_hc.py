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


def test_get_component_weights_after_fit():
    """get_component_weights returns one weight per component, all in [0, 1]."""
    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=24, n_labels=2)
    hc2 = HIVECOTEV2(**HIVECOTEV2._get_test_params(parameter_set="default"))
    hc2.fit(X, y)

    weights = hc2.get_component_weights()
    assert set(weights.keys()) == {"STC", "DrCIF", "Arsenal", "TDE"}
    for name, w in weights.items():
        assert 0.0 <= w <= 1.0, f"weight for {name} out of range: {w}"


def test_base_rejects_non_baseclassifier():
    """_BaseHIVECOTE._fit raises TypeError for non-BaseClassifier components."""
    from aeon.classification.hybrid._base_hive_cote import _BaseHIVECOTE

    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=24, n_labels=2)

    class NotAClassifier:
        def fit_predict(self, X, y):
            return y

    clf = _BaseHIVECOTE(estimators=[("bad", NotAClassifier())])
    with pytest.raises(TypeError, match="not a BaseClassifier"):
        clf.fit(X, y)


def test_base_rejects_empty_estimators():
    """_BaseHIVECOTE._fit raises ValueError for empty or None estimators."""
    from aeon.classification.hybrid._base_hive_cote import _BaseHIVECOTE

    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=24, n_labels=2)
    clf = _BaseHIVECOTE(estimators=[])
    with pytest.raises(ValueError, match="No estimators provided"):
        clf.fit(X, y)


@pytest.mark.skipif(PR_TESTING, reason="slow test, run overnight only")
def test_refit_resets_state():
    """Re-fitting resets fitted state (no accumulation)."""
    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=24, n_labels=2)
    hc2 = HIVECOTEV2(**HIVECOTEV2._get_test_params(parameter_set="default"))

    hc2.fit(X, y)
    assert len(hc2.fitted_estimators_) == 4
    assert len(hc2.weights_) == 4
    assert len(hc2.component_names_) == 4

    hc2.fit(X, y)
    assert (
        len(hc2.fitted_estimators_) == 4
    ), f"fitted_estimators_ accumulated on re-fit: got {len(hc2.fitted_estimators_)}"
    assert len(hc2.weights_) == 4
    assert len(hc2.component_names_) == 4


def test_weight_property_returns_zero_before_fit():
    """Weight properties return 0.0 before fit (not AttributeError)."""
    hc2 = HIVECOTEV2()
    assert hc2.stc_weight_ == 0.0
    assert hc2.drcif_weight_ == 0.0
    assert hc2.arsenal_weight_ == 0.0
    assert hc2.tde_weight_ == 0.0

    hc1 = HIVECOTEV1()
    assert hc1.stc_weight_ == 0.0
    assert hc1.tsf_weight_ == 0.0
    assert hc1.rise_weight_ == 0.0
    assert hc1.cboss_weight_ == 0.0
