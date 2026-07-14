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


def test_hc2_default_component_parameters(monkeypatch):
    """HC2 constructs its four components from the documented default sizes."""
    default_sizes = {
        "_DEFAULT_N_TREES": 1,
        "_DEFAULT_N_SHAPELETS": 5,
        "_DEFAULT_N_KERNELS": 5,
        "_DEFAULT_N_ESTIMATORS": 1,
        "_DEFAULT_N_PARA_SAMPLES": 1,
        "_DEFAULT_MAX_ENSEMBLE_SIZE": 1,
        "_DEFAULT_RAND_PARAMS": 1,
    }
    for attribute, value in default_sizes.items():
        monkeypatch.setattr(HIVECOTEV2, attribute, value)

    X, y = make_example_3d_numpy(
        n_cases=20, n_timepoints=20, n_labels=2, random_state=0
    )
    hc2 = HIVECOTEV2(random_state=0)
    hc2.fit(X, y)

    components = dict(zip(hc2.component_names_, hc2.fitted_estimators_))
    assert components["STC"].n_shapelet_samples == 5
    assert components["DrCIF"].n_estimators == 1
    assert components["Arsenal"].n_kernels == 5
    assert components["Arsenal"].n_estimators == 1
    assert components["TDE"].n_parameter_samples == 1
    assert components["TDE"].max_ensemble_size == 1
    assert components["TDE"].randomly_selected_params == 1


def test_hc2_verbose_progress_and_parameter_output(capsys):
    """HC2 verbosity two reports ensemble progress and component parameters."""
    n_cases = 20
    n_timepoints = 24
    X, y = make_example_3d_numpy(n_cases=n_cases, n_timepoints=n_timepoints, n_labels=2)
    hc2 = HIVECOTEV2(
        verbose=2,
        **HIVECOTEV2._get_test_params(parameter_set="default"),
    )

    hc2.fit(X, y)
    output = capsys.readouterr().out

    assert f"[HC2] Starting fit: n_cases={n_cases}" in output
    for component_name in ("STC", "DrCIF", "Arsenal", "TDE"):
        assert f"[HC2] Starting {component_name}..." in output
        assert f"[HC2] {component_name} params:" in output
        assert f"[HC2] Finished {component_name} in " in output
    assert "[HC2] Finished fit in " in output
    assert "[HC2] Component summary:" in output


def test_hc2_contract_allocation_is_logged_without_mutating_params(capsys):
    """HC2 reports its contract split without changing caller-owned dictionaries."""
    contract_minutes = 0.01
    params = HIVECOTEV2._get_test_params(parameter_set="contracting")
    params["time_limit_in_minutes"] = contract_minutes
    component_param_names = (
        "stc_params",
        "drcif_params",
        "arsenal_params",
        "tde_params",
    )
    X, y = make_example_3d_numpy(
        n_cases=20, n_timepoints=24, n_labels=2, random_state=0
    )

    hc2 = HIVECOTEV2(verbose=1, random_state=0, **params).fit(X, y)
    output = capsys.readouterr().out

    component_minutes = contract_minutes / 6
    assert "[HC2] Contract time = 0.01 minutes" in output
    components = dict(zip(hc2.component_names_, hc2.fitted_estimators_))
    for param_name in component_param_names:
        assert "time_limit_in_minutes" not in params[param_name]
    assert all(
        component.time_limit_in_minutes == component_minutes
        for component in components.values()
    )


def test_get_component_weights_after_fit():
    """get_component_weights returns one weight per component, all in [0, 1]."""
    X, y = make_example_3d_numpy(n_cases=20, n_timepoints=24, n_labels=2)
    hc2 = HIVECOTEV2(**HIVECOTEV2._get_test_params(parameter_set="default"))
    hc2.fit(X, y)

    weights = hc2.get_component_weights()
    assert set(weights.keys()) == {"STC", "DrCIF", "Arsenal", "TDE"}
    for name, w in weights.items():
        assert 0.0 <= w <= 1.0, f"weight for {name} out of range: {w}"


def test_alpha_controls_component_weight_exponent():
    """HC2 alpha controls the exponent applied to component train accuracy."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_timepoints=24, n_labels=2, random_state=0
    )
    params = HIVECOTEV2._get_test_params(parameter_set="default")
    alpha_two = HIVECOTEV2(alpha=2, random_state=0, **params).fit(X, y)
    alpha_four = HIVECOTEV2(alpha=4, random_state=0, **params).fit(X, y)

    weights_two = np.asarray(alpha_two.weights_)
    weights_four = np.asarray(alpha_four.weights_)
    np.testing.assert_allclose(weights_four, weights_two**2)


def test_component_probabilities_and_deprecated_storage():
    """HC2 returns named component probabilities and supports deprecated storage."""
    n_cases = 20
    X, y = make_example_3d_numpy(
        n_cases=n_cases, n_timepoints=24, n_labels=2, random_state=0
    )
    hc2 = HIVECOTEV2(
        random_state=0,
        **HIVECOTEV2._get_test_params(parameter_set="results_comparison"),
    ).fit(X, y)
    expected = hc2.predict_proba(X)

    combined, components = hc2.predict_proba_with_components(X)

    np.testing.assert_allclose(combined, expected)
    assert set(components) == {"STC", "DrCIF", "Arsenal", "TDE"}
    assert all(probas.shape == expected.shape for probas in components.values())
    assert not hasattr(hc2, "component_probas")

    hc2.save_component_probas = True
    with pytest.warns(DeprecationWarning, match="mutates object state"):
        stored = hc2.predict_proba(X)

    np.testing.assert_allclose(stored, expected)
    for component_name, probas in components.items():
        np.testing.assert_allclose(hc2.component_probas[component_name], probas)


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
