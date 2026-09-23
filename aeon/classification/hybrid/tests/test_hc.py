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


def test_hc1_verbose_progress_and_parameter_output(capsys):
    """HC1 verbosity four reports ensemble and detailed component progress."""
    n_cases = 20
    n_timepoints = 24
    X, y = make_example_3d_numpy(n_cases=n_cases, n_timepoints=n_timepoints, n_labels=2)
    hc1 = HIVECOTEV1(
        verbose=4,
        **HIVECOTEV1._get_test_params(parameter_set="default"),
    )

    hc1.fit(X, y)
    output = capsys.readouterr().out

    assert f"[HIVECOTEV1] Starting fit: n_cases={n_cases}" in output
    for component_name in ("STC", "TSF", "RISE", "cBOSS"):
        assert f"[HIVECOTEV1] Starting {component_name}..." in output
        assert f"[HIVECOTEV1] {component_name} params:" in output
        assert f"[HIVECOTEV1] Finished {component_name} in " in output
    assert "[HIVECOTEV1] Finished fit in " in output
    assert "[HIVECOTEV1] Component summary:" in output
    assert "[ShapeletTransformClassifier] Starting fit:" in output
    assert "[RandomShapeletTransform] Batch 1:" in output

    components = dict(zip(hc1.component_names_, hc1.fitted_estimators_))
    assert components["STC"].verbose == 2
    assert components["STC"].transformer_.verbose == 2


def test_hc2_verbose_progress_and_parameter_output(capsys):
    """HC2 verbosity four reports ensemble and detailed component progress."""
    n_cases = 20
    n_timepoints = 24
    X, y = make_example_3d_numpy(n_cases=n_cases, n_timepoints=n_timepoints, n_labels=2)
    hc2 = HIVECOTEV2(
        verbose=4,
        **HIVECOTEV2._get_test_params(parameter_set="default"),
    )

    hc2.fit(X, y)
    output = capsys.readouterr().out

    assert f"[HIVECOTEV2] Starting fit: n_cases={n_cases}" in output
    for component_name in ("STC", "DrCIF", "Arsenal", "TDE"):
        assert f"[HIVECOTEV2] Starting {component_name}..." in output
        assert f"[HIVECOTEV2] {component_name} params:" in output
        assert f"[HIVECOTEV2] Finished {component_name} in " in output
    assert "[HIVECOTEV2] Finished fit in " in output
    assert "[HIVECOTEV2] Component summary:" in output
    assert "[RandomShapeletTransform] Batch 1:" in output
    assert "[DrCIFClassifier] Estimator 1/" in output
    assert "[Arsenal] Estimator 1/" in output
    assert "[TemporalDictionaryEnsemble] Candidate 1:" in output

    components = dict(zip(hc2.component_names_, hc2.fitted_estimators_))
    assert components["STC"].verbose == 2
    assert components["STC"].transformer_.verbose == 2
    # STC only passes verbosity to RotationForestClassifier, not scikit-learn
    assert components["STC"].estimator_.verbose == 0
    assert components["DrCIF"].verbose == 2
    assert components["Arsenal"].verbose == 2
    assert components["TDE"].verbose == 2


def test_hc2_contract_allocation_is_logged(capsys):
    """HC2 reports how its contract is split between components."""
    contract_minutes = 0.01
    params = HIVECOTEV2._get_test_params(parameter_set="contracting")
    params["time_limit_in_minutes"] = contract_minutes
    X, y = make_example_3d_numpy(
        n_cases=20, n_timepoints=24, n_labels=2, random_state=0
    )

    HIVECOTEV2(verbose=1, random_state=0, **params).fit(X, y)
    output = capsys.readouterr().out

    assert "[HIVECOTEV2] Contract time = 0.01 minutes" in output
    assert "per-component allocation = 0.0017 minutes" in output


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


def test_hivecote_estimator_attribute_lifecycle():
    """Test estimator attributes are created only on fit."""
    from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
    from aeon.testing.data_generation import make_example_3d_numpy

    X, y = make_example_3d_numpy(
        n_cases=10,
        n_channels=1,
        n_timepoints=20,
        min_cases_per_label=3,
        random_state=0,
    )

    for hc_class in [HIVECOTEV1, HIVECOTEV2]:
        params = hc_class._get_test_params()
        if isinstance(params, list):
            params = params[0]

        clf = hc_class(**params)

        assert not hasattr(clf, "fitted_estimators_")

        clf.fit(X, y)

        assert hasattr(clf, "fitted_estimators_")
