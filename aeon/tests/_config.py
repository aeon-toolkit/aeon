__author__ = ["mloning"]
__all__ = ["EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

import os
import pickle

import numba.core.caching

from aeon.base import BaseEstimator, BaseObject
from aeon.registry import BASE_CLASS_LIST, BASE_CLASS_LOOKUP, ESTIMATOR_TAG_LIST

DEBUG_CACHE = True


def custom_load_index(self):
    try:
        with open(self._index_path, "rb") as f:
            version = pickle.load(f)
            data = f.read()
    except FileNotFoundError:
        # Index doesn't exist yet?
        return {}
    if version != self._version:
        # This is another version.  Avoid trying to unpickling the
        # rest of the stream, as that may fail.
        return {}
    stamp, overloads = pickle.loads(data)
    if DEBUG_CACHE:
        print("[cache] index loaded from %r", self._index_path)  # noqa: T001 T201
    if stamp[1] != self._source_stamp[1]:
        # Cache is not fresh.  Stale data files will be eventually
        # overwritten, since they are numbered in incrementing order.
        return {}
    else:
        return overloads


original_load_index = numba.core.caching.IndexDataCacheFile._load_index

numba.core.caching.IndexDataCacheFile._load_index = custom_load_index

# whether to use smaller parameter matrices for test generation and subsample estimators
# per os/version default is False, can be set to True by pytest --prtesting True flag
PR_TESTING = False

EXCLUDE_ESTIMATORS = []

# the test currently fails when numba is disabled. See issue #622
if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    EXCLUDE_ESTIMATORS.append("StatsForecastAutoARIMA")

EXCLUDED_TESTS = {
    # Early classifiers (EC) intentionally retain information from previous predict
    # calls for #1 (test_non_state_changing_method_contract).
    # #2 (test_fit_deterministic), #3 (test_persistence_via_pickle) and #4
    # (test_save_estimators_to_file) are due to predict/predict_proba returning two
    # items and that breaking assert_array_equal.
    "TEASER": [  # EC
        "test_non_state_changing_method_contract",
        "test_fit_deterministic",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    "ProbabilityThresholdEarlyClassifier": [  # EC
        "test_non_state_changing_method_contract",
        "test_fit_deterministic",
        "test_persistence_via_pickle",
        "test_save_estimators_to_file",
    ],
    # GGS inherits from BaseEstimator which breaks this test: see
    # https://github.com/aeon-toolkit/aeon/issues/699
    "GreedyGaussianSegmentation": ["test_inheritance", "test_create_test_instance"],
    "InformationGainSegmentation": [
        "test_inheritance",
        "test_create_test_instance",
    ],
    # test fails several variants of inversion, see
    # https://github.com/aeon-toolkit/aeon/issues/700
    "Differencer": ["test_transform_inverse_transform_equivalent"],
}

# We use estimator tags in addition to class hierarchies to further distinguish
# estimators into different categories. This is useful for defining and running
# common tests for estimators with the same tags.
VALID_ESTIMATOR_TAGS = tuple(ESTIMATOR_TAG_LIST)

# NON_STATE_CHANGING_METHODS =
# methods that should not change the state of the estimator, that is, they should
# not change fitted parameters or hyper-parameters. They are also the methods that
# "apply" the fitted estimator to data and useful for checking results.
# NON_STATE_CHANGING_METHODS_ARRAYLIK =
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
BASE_BASE_TYPES = (BaseEstimator, BaseObject)
VALID_ESTIMATOR_BASE_TYPES = tuple(set(BASE_CLASS_LIST).difference(BASE_BASE_TYPES))

VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
)

VALID_ESTIMATOR_BASE_TYPE_LOOKUP = BASE_CLASS_LOOKUP
