"""Test configuration."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import aeon.testing.utils._cicd_numba_caching  # noqa: F401
from aeon.base import BaseAeonEstimator, BaseCollectionEstimator, BaseSeriesEstimator
from aeon.registry import BASE_CLASS_LIST, BASE_CLASS_LOOKUP

# whether to use smaller parameter matrices for test generation and subsample estimators
# per os/version default is False, can be set to True by pytest --prtesting True flag
PR_TESTING = False

# Exclude estimators here for short term fixes
EXCLUDE_ESTIMATORS = [
    "ClearSkyTransformer",
    # See #2071
    "RISTRegressor",
]


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
    # has a keras fail, unknown reason, see #1387
    "LearningShapeletClassifier": ["check_fit_deterministic"],
    # does not fit structure for test, needs investigation
    "TapNetClassifier": ["check_classifier_random_state_deep_learning"],
    "TapNetRegressor": ["check_regressor_random_state_deep_learning"],
    # needs investigation
    "SASTClassifier": ["check_fit_deterministic"],
    "RSASTClassifier": ["check_fit_deterministic"],
    "SAST": ["check_fit_deterministic"],
    "RSAST": ["check_fit_deterministic"],
    "AEFCNClusterer": ["check_fit_updates_state"],
    "AEResNetClusterer": ["check_fit_updates_state"],
    "SFA": ["check_persistence_via_pickle", "check_fit_deterministic"],
    "CollectionId": ["check_transform_inverse_transform_equivalent"],
    "ScaledLogitSeriesTransformer": ["check_transform_inverse_transform_equivalent"],
    # also uncomment in test_check_estimator.py
    "MockMultivariateSeriesTransformer": [
        "check_transform_inverse_transform_equivalent"
    ],
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

# The following gives a list of valid estimator base classes.
CORE_BASE_TYPES = (
    BaseAeonEstimator,
    BaseCollectionEstimator,
    BaseSeriesEstimator,
)
VALID_ESTIMATOR_BASE_TYPES = tuple(set(BASE_CLASS_LIST).difference(CORE_BASE_TYPES))

VALID_ESTIMATOR_TYPES = (
    BaseAeonEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = BASE_CLASS_LOOKUP
