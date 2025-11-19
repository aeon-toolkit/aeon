"""Test configuration."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "PR_TESTING",
    "EXCLUDE_ESTIMATORS",
    "EXCLUDED_TESTS",
    "EXCLUDED_TESTS_NO_NUMBA",
]

import os

import aeon.testing._cicd_numba_caching  # noqa: F401

# whether to use smaller parameter matrices for test generation and subsample estimators
# per os/version default is False, can be set to True by pytest --prtesting True flag
PR_TESTING = False
# whether to use multithreading in tests, can be set to True by pytest
# --enablethreading True flag
MULTITHREAD_TESTING = False

# whether numba is disabled vis environment variable
NUMBA_DISABLED = os.environ.get("NUMBA_DISABLE_JIT") == "1"

# exclude estimators here for short term fixes
EXCLUDE_ESTIMATORS = [
    "HydraTransformer",  # returns a pytorch Tensor
]

# Exclude specific tests for estimators here
EXCLUDED_TESTS = {
    # Early classifiers (EC) intentionally retain information from previous predict
    # calls for #1 (test_non_state_changing_method_contract).
    # #2 (test_fit_deterministic), #3 (test_persistence_via_pickle) and #4
    # (test_save_estimators_to_file) are due to predict/predict_proba returning two
    # items and that breaking assert_array_equal.
    "TEASER": [  # EC
        "check_non_state_changing_method",
        "check_fit_deterministic",
        "check_persistence_via_pickle",
        "check_save_estimators_to_file",
    ],
    "ProbabilityThresholdEarlyClassifier": [  # EC
        "check_non_state_changing_method",
        "check_fit_deterministic",
        "check_persistence_via_pickle",
        "check_save_estimators_to_file",
    ],
    # needs investigation
    "LeftSTAMPi": ["check_series_anomaly_detector_output"],
    # missed in legacy testing, changes state in predict/transform
    "FLUSSSegmenter": ["check_non_state_changing_method"],
    "ClaSPSegmenter": ["check_non_state_changing_method"],
    "HMMSegmenter": ["check_non_state_changing_method"],
    # Unknown issue not producing the same results
    "RDSTRegressor": ["check_regressor_against_expected_results"],
    "RISTRegressor": ["check_regressor_against_expected_results"],
    # Affected by threading changes in distance module
    "CanonicalIntervalForestRegressor": ["check_regressor_against_expected_results"],
    # Requires y to be passed in inverse_transform,
    # but this is not currently enabled/supported
    "DifferenceTransformer": ["check_transform_inverse_transform_equivalent"],
}

# Exclude specific tests for estimators here only when numba is disabled
EXCLUDED_TESTS_NO_NUMBA = {
    # See issue #622
    "HIVECOTEV2": ["check_classifier_against_expected_results"],
    # Other failures
    "TemporalDictionaryEnsemble": ["check_classifier_against_expected_results"],
    "OrdinalTDE": ["check_classifier_against_expected_results"],
    "MultiRocketRegressor": ["check_regressor_against_expected_results"],
    "MultiRocketHydraRegressor": ["check_regressor_against_expected_results"],
}


# estimator methods post-fit that should not change the state of the estimator

# non-state-changing methods that return an array-like output
NON_STATE_CHANGING_METHODS_ARRAYLIKE = (
    "predict",
    "predict_proba",
    "transform",
)

# all non-state-changing methods
NON_STATE_CHANGING_METHODS = NON_STATE_CHANGING_METHODS_ARRAYLIKE + (
    "get_fitted_params",
)
