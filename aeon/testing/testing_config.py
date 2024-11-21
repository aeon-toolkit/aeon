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

# whether numba is disabled vis environment variable
NUMBA_DISABLED = os.environ.get("NUMBA_DISABLE_JIT") == "1"

# Exclude estimators here for short term fixes
EXCLUDE_ESTIMATORS = []

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
    "SASTClassifier": ["check_fit_deterministic"],
    "RSASTClassifier": ["check_fit_deterministic"],
    "SAST": ["check_fit_deterministic"],
    "RSAST": ["check_fit_deterministic"],
    "SFA": ["check_persistence_via_pickle", "check_fit_deterministic"],
    # missed in legacy testing, changes state in predict/transform
    "FLUSSSegmenter": ["check_non_state_changing_method"],
    "InformationGainSegmenter": ["check_non_state_changing_method"],
    "GreedyGaussianSegmenter": ["check_non_state_changing_method"],
    "ClaSPSegmenter": ["check_non_state_changing_method"],
    "HMMSegmenter": ["check_non_state_changing_method"],
    "BinSegSegmenter": ["check_non_state_changing_method"],
    "QUANTTransformer": ["check_non_state_changing_method"],
    "MatrixProfileSeriesTransformer": ["check_non_state_changing_method"],
    "PLASeriesTransformer": ["check_non_state_changing_method"],
    "AutoCorrelationSeriesTransformer": ["check_non_state_changing_method"],
    "SIVSeriesTransformer": ["check_non_state_changing_method"],
    "RocketClassifier": ["check_non_state_changing_method"],
    "MiniRocketClassifier": ["check_non_state_changing_method"],
    "MultiRocketClassifier": ["check_non_state_changing_method"],
    "RocketRegressor": ["check_non_state_changing_method"],
    "MiniRocketRegressor": ["check_non_state_changing_method"],
    "MultiRocketRegressor": ["check_non_state_changing_method"],
    "RSTSF": ["check_non_state_changing_method"],
    # Keeps length during predict to avoid recomputing means and std of data in fit
    # if the next predict calls uses the same query length parameter.
    "QuerySearch": ["check_non_state_changing_method"],
    "SeriesSearch": ["check_non_state_changing_method"],
    # Unknown issue not producing the same results
    "RDSTRegressor": ["check_regressor_against_expected_results"],
    "RISTRegressor": ["check_regressor_against_expected_results"],
}

# Exclude specific tests for estimators here only when numba is disabled
EXCLUDED_TESTS_NO_NUMBA = {
    # See issue #622
    "HIVECOTEV2": ["check_classifier_against_expected_results"],
}

# NON_STATE_CHANGING_METHODS =
# methods that should not change the state of the estimator, that is, they should
# not change fitted parameters or hyper-parameters. They are also the methods that
# "apply" the fitted estimator to data and useful for checking results.

# non-state-changing methods that return an array-like output
NON_STATE_CHANGING_METHODS_ARRAYLIKE = (
    "predict",
    "predict_var",
    "predict_proba",
    "decision_function",
    "transform",
)

NON_STATE_CHANGING_METHODS = NON_STATE_CHANGING_METHODS_ARRAYLIKE + (
    "get_fitted_params",
)
