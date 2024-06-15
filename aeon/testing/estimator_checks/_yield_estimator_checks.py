import numbers
import pickle
import types
from copy import deepcopy
from functools import partial
from inspect import getfullargspec, signature

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_get_params_invariance

from aeon.base import BaseEstimator, BaseObject
from aeon.base._base import _clone_estimator
from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.clustering.deep_learning.base import BaseDeepClusterer
from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.testing.test_config import (
    NON_STATE_CHANGING_METHODS,
    NON_STATE_CHANGING_METHODS_ARRAYLIKE,
    VALID_ESTIMATOR_BASE_TYPES,
    VALID_ESTIMATOR_TAGS,
)
from aeon.testing.testing_data import (
    TEST_DATA_DICT,
    TEST_LABEL_DICT,
    get_data_types_for_estimator,
)
from aeon.testing.utils.deep_equals import deep_equals
from aeon.testing.utils.estimator_checks import (
    _assert_array_almost_equal,
    _get_args,
    _list_required_methods,
    _run_estimator_method,
)
from aeon.transformations.base import BaseTransformer


def _yield_all_aeon_checks(estimator):
    datatypes = get_data_types_for_estimator(estimator)

    yield from _yield_estimator_checks(estimator, datatypes)


def _yield_estimator_checks(estimator, datatypes):
    # no data needed
    yield check_create_test_instance
    yield check_create_test_instances_and_names
    yield check_estimator_tags
    yield check_inheritance
    yield check_has_common_interface
    yield check_get_params
    yield check_set_params
    yield check_set_params_sklearn
    yield check_clone
    yield check_repr
    yield check_constructor
    yield check_valid_estimator_class_tags
    yield check_valid_estimator_tags

    if (
        isinstance(estimator, BaseDeepClassifier)
        or isinstance(estimator, BaseDeepRegressor)
        or isinstance(estimator, BaseDeepClusterer)
    ):
        yield check_dl_constructor_initializes_deeply

    # data type irrelevant
    yield partial(check_non_state_changing_method, datatype=datatypes[0])
    yield partial(check_fit_updates_state, datatype=datatypes[0])

    if not estimator.get_tag(
        "fit_is_empty", tag_value_default=False, raise_error=False
    ):
        yield partial(check_raises_not_fitted_error, datatype=datatypes[0])

    if not estimator.get_tag("cant-pickle", tag_value_default=False, raise_error=False):
        yield partial(test_persistence_via_pickle, datatype=datatypes[0])

    if not estimator.get_tag(
        "non-deterministic", tag_value_default=False, raise_error=False
    ):
        yield partial(check_fit_deterministic, datatype=datatypes[0])


def check_create_test_instance(estimator):
    """Check create_test_instance logic and basic constructor functionality.

    create_test_instance and create_test_instances_and_names are the
    key methods used to create test instances in testing.
    If this test does not pass, validity of the other tests cannot be guaranteed.

    Also tests inheritance and super call logic in the constructor.

    Tests that:
    * create_test_instance results in an instance of estimator_class
    * __init__ calls super.__init__
    * _tags_dynamic attribute for tag inspection is present after construction
    """
    estimator_class = type(estimator)
    estimator = estimator_class.create_test_instance()

    # Check that method does not construct object of other class than itself
    assert isinstance(estimator, estimator_class), (
        "object returned by create_test_instance must be an instance of the class, "
        f"found {type(estimator)}"
    )

    msg = (
        f"{estimator_class.__name__}.__init__ should call super().__init__, the "
        "estimator does not produce the attributes this call would produce."
    )
    assert hasattr(estimator, "_tags_dynamic"), msg


# todo consider deprecation
def check_create_test_instances_and_names(estimator):
    """Check that create_test_instances_and_names works.

    create_test_instance and create_test_instances_and_names are the
    key methods used to create test instances in testing.
    If this test does not pass, validity of the other tests cannot be guaranteed.

    Tests expected function signature of create_test_instances_and_names.
    """
    estimator_class = type(estimator)
    estimators, names = estimator_class.create_test_instances_and_names()

    assert isinstance(estimators, list), (
        "first return of create_test_instances_and_names must be a list, "
        f"found {type(estimators)}"
    )
    assert isinstance(names, list), (
        "second return of create_test_instances_and_names must be a list, "
        f"found {type(names)}"
    )

    assert np.all([isinstance(est, estimator_class) for est in estimators]), (
        "list elements of first return returned by create_test_instances_and_names "
        "all must be an instance of the class"
    )

    assert np.all([isinstance(name, str) for name in names]), (
        "list elements of second return returned by create_test_instances_and_names"
        " all must be strings"
    )

    assert len(estimators) == len(names), (
        "the two lists returned by create_test_instances_and_names must have "
        "equal length"
    )


# todo consider expanding to init and compare against registry classes
def check_estimator_tags(estimator):
    """Check conventions on estimator tags."""
    estimator_class = type(estimator)

    assert hasattr(estimator_class, "get_class_tags")
    all_tags = estimator_class.get_class_tags()
    assert isinstance(all_tags, dict)
    assert all(isinstance(key, str) for key in all_tags.keys())
    if hasattr(estimator_class, "_tags"):
        tags = estimator_class._tags
        msg = (
            f"_tags attribute of {estimator_class} must be dict, "
            f"but found {type(tags)}"
        )
        assert isinstance(tags, dict), msg
        assert len(tags) > 0, f"_tags dict of class {estimator_class} is empty"
        invalid_tags = [tag for tag in tags.keys() if tag not in VALID_ESTIMATOR_TAGS]
        assert len(invalid_tags) == 0, (
            f"_tags of {estimator_class} contains invalid tags: {invalid_tags}. "
            "For a list of valid tags, see registry.all_tags, or registry._tags. "
        )

    # Avoid ambiguous class attributes
    ambiguous_attrs = ("tags", "tags_")
    for attr in ambiguous_attrs:
        assert not hasattr(estimator_class, attr), (
            f"Please avoid using the {attr} attribute to disambiguate it from "
            f"estimator tags."
        )


# todo consider removing the multiple base class allowance. Possibly deprecate
#  BaseObject and roll it into BaseEstimator?
def check_inheritance(estimator):
    estimator_class = type(estimator)

    """Check that estimator inherits from BaseObject and/or BaseEstimator."""
    assert issubclass(
        estimator_class, BaseObject
    ), f"object {estimator_class} is not a sub-class of BaseObject."

    if hasattr(estimator_class, "fit"):
        assert issubclass(estimator_class, BaseEstimator), (
            f"estimator: {estimator_class} has fit method, but"
            f"is not a sub-class of BaseEstimator."
        )

    # Usually estimators inherit only from one BaseEstimator type, but in some cases
    # they may be predictor and transformer at the same time (e.g. pipelines)
    n_base_types = sum(
        issubclass(estimator_class, cls) for cls in VALID_ESTIMATOR_BASE_TYPES
    )

    assert 2 >= n_base_types >= 1

    # If the estimator inherits from more than one base estimator type, we check if
    # one of them is a transformer base type
    if n_base_types > 1:
        assert issubclass(estimator_class, BaseTransformer)


def check_has_common_interface(estimator):
    """Check estimator implements the common interface."""
    estimator_class = type(estimator)

    # Check class for type of attribute
    if isinstance(estimator_class, BaseEstimator):
        assert isinstance(estimator_class.is_fitted, property)

    required_methods = _list_required_methods(estimator_class)

    for attr in required_methods:
        assert hasattr(
            estimator_class, attr
        ), f"Estimator: {estimator_class.__name__} does not implement attribute: {attr}"

    if hasattr(estimator_class, "inverse_transform"):
        assert hasattr(estimator_class, "transform")
    if hasattr(estimator_class, "predict_proba"):
        assert hasattr(estimator_class, "predict")


def check_get_params(estimator):
    """Check that get_params works correctly."""
    params = estimator.get_params()
    assert isinstance(params, dict)
    check_get_params_invariance(estimator.__class__.__name__, estimator)


def check_set_params(estimator):
    """Check that set_params works correctly."""
    params = estimator.get_params()

    msg = f"set_params of {type(estimator).__name__} does not return self"
    assert estimator.set_params(**params) is estimator, msg

    is_equal, equals_msg = deep_equals(estimator.get_params(), params, return_msg=True)
    msg = (
        f"get_params result of {type(estimator).__name__} (x) does not match "
        f"what was passed to set_params (y). Reason for discrepancy: {equals_msg}"
    )
    assert is_equal, msg


def check_set_params_sklearn(estimator):
    """Check that set_params works correctly, mirrors sklearn check_set_params.

    Instead of the "fuzz values" in sklearn's check_set_params,
    we use the other test parameter settings (which are assumed valid).
    This guarantees settings which play along with the __init__ content.
    """
    estimator_class = type(estimator)

    estimator = estimator_class.create_test_instance()
    test_params = estimator_class.get_test_params()
    if not isinstance(test_params, list):
        test_params = [test_params]

    for params in test_params:
        # we construct the full parameter set for params
        # params may only have parameters that are deviating from defaults
        # in order to set non-default parameters back to defaults
        params_full = estimator_class.get_param_defaults()
        params_full.update(params)

        msg = f"set_params of {estimator_class.__name__} does not return self"
        est_after_set = estimator.set_params(**params_full)
        assert est_after_set is estimator, msg

        is_equal, equals_msg = deep_equals(
            estimator.get_params(deep=False), params_full, return_msg=True
        )
        msg = (
            f"get_params result of {estimator_class.__name__} (x) does not match "
            f"what was passed to set_params (y). "
            f"Reason for discrepancy: {equals_msg}"
        )
        assert is_equal, msg


def check_clone(estimator):
    """Check that clone method does not raise exceptions and results in a clone.

    A clone of an object x is an object that:
    * has same class and parameters as x
    * is not identical with x
     * is unfitted (even if x was fitted)
    """
    est_clone = estimator.clone()
    assert isinstance(est_clone, type(estimator))
    assert est_clone is not estimator
    if hasattr(est_clone, "is_fitted"):
        assert not est_clone.is_fitted


# todo roll into another test
def check_repr(estimator):
    """Check that __repr__ call to instance does not raise exceptions."""
    repr(estimator)


def check_constructor(estimator):
    """Check that the constructor has sklearn compatible signature and behaviour.

    Based on sklearn check_estimator testing of __init__ logic.
    Uses create_test_instance to create an instance.
    Assumes test_create_test_instance has passed and certified create_test_instance.

    Tests that:
    * constructor has no varargs
    * tests that constructor constructs an instance of the class
    * tests that all parameters are set in init to an attribute of the same name
    * tests that parameter values are always copied to the attribute and not changed
    * tests that default parameters are one of the following:
        None, str, int, float, bool, tuple, function, joblib memory, numpy primitive
        (other type parameters should be None, default handling should be by writing
        the default to attribute of a different name, e.g., my_param_ not my_param)
    """
    estimator_class = type(estimator)

    msg = "constructor __init__ should have no varargs"
    assert getfullargspec(estimator_class.__init__).varkw is None, msg

    estimator = estimator_class.create_test_instance()
    assert isinstance(estimator, estimator_class)

    # Ensure that each parameter is set in init
    init_params = _get_args(type(estimator).__init__)
    invalid_attr = set(init_params) - set(vars(estimator)) - {"self"}
    assert not invalid_attr, (
        "Estimator %s should store all parameters"
        " as an attribute during init. Did not find "
        "attributes `%s`." % (estimator.__class__.__name__, sorted(invalid_attr))
    )

    # Ensure that init does nothing but set parameters
    # No logic/interaction with other parameters
    def param_filter(p):
        """Identify hyper parameters of an estimator."""
        return p.name != "self" and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]

    init_params = [
        p for p in signature(estimator.__init__).parameters.values() if param_filter(p)
    ]

    params = estimator.get_params()

    test_params = estimator_class.get_test_params()
    if isinstance(test_params, list):
        test_params = test_params[0]
    test_params = test_params.keys()

    init_params = [param for param in init_params if param.name not in test_params]

    for param in init_params:
        assert param.default != param.empty, (
            "parameter `%s` for %s has no default value and is not "
            "set in `get_test_params`" % (param.name, estimator.__class__.__name__)
        )
        if type(param.default) is type:
            assert param.default in [np.float64, np.int64]
        else:
            assert type(param.default) in [
                str,
                int,
                float,
                bool,
                tuple,
                type(None),
                np.float64,
                types.FunctionType,
                joblib.Memory,
            ]

        param_value = params[param.name]
        if isinstance(param_value, np.ndarray):
            np.testing.assert_array_equal(param_value, param.default)
        else:
            if bool(isinstance(param_value, numbers.Real) and np.isnan(param_value)):
                # Allows to set default parameters to np.nan
                assert param_value is param.default, param.name
            else:
                assert param_value == param.default, param.name


def check_valid_estimator_class_tags(estimator):
    """Check that Estimator class tags are in VALID_ESTIMATOR_TAGS."""
    estimator_class = type(estimator)
    for tag in estimator_class.get_class_tags().keys():
        assert tag in VALID_ESTIMATOR_TAGS


def check_valid_estimator_tags(estimator):
    """Check that Estimator tags are in VALID_ESTIMATOR_TAGS."""
    for tag in estimator.get_tags().keys():
        assert tag in VALID_ESTIMATOR_TAGS


def check_dl_constructor_initializes_deeply(estimator):
    """Test DL estimators that they pass custom parameters to underlying Network."""
    if not hasattr(estimator, "get_test_params"):
        return None

    params = estimator.get_test_params()

    if isinstance(params, list):
        params = params[0]
    if isinstance(params, dict):
        pass
    else:
        raise TypeError(
            f"`get_test_params()` of estimator: {estimator} returns "
            f"an expected type: {type(params)}, acceptable formats: [list, dict]"
        )

    estimator = estimator(**params)

    for key, value in params.items():
        assert vars(estimator)[key] == value
        # some keys are only relevant to the final model (eg: n_epochs)
        # skip them for the underlying network
        if vars(estimator._network).get(key) is not None:
            assert vars(estimator._network)[key] == value


def check_non_state_changing_method(estimator, datatype):
    """Check that non-state-changing methods behave as per interface contract.

    Check the following contract on non-state-changing methods:
    1. do not change state of the estimator, i.e., any attributes
        (including hyper-parameters and fitted parameters)
    2. expected output type of the method matches actual output type
        - only for abstract BaseEstimator methods, common to all estimators.
        List of BaseEstimator methods tested: get_fitted_params
        Subclass specific method outputs are tested in TestAll[estimatortype] class
    3. the state of method arguments does not change
    """
    estimator = _clone_estimator(estimator)

    X = deepcopy(TEST_DATA_DICT[datatype[0]]["train"])
    y = deepcopy(TEST_LABEL_DICT[datatype[1]]["train"])
    _run_estimator_method(estimator, "fit", datatype, "train")

    assert deep_equals(X, TEST_DATA_DICT[datatype[0]]["train"]) and deep_equals(
        y, TEST_LABEL_DICT[datatype[1]]["train"]
    ), f"Estimator: {type(estimator)} has side effects on arguments of fit"

    # dict_before = copy of dictionary of estimator before predict, post fit
    dict_before = estimator.__dict__.copy()
    X = deepcopy(TEST_DATA_DICT[datatype[0]]["test"])
    y = deepcopy(TEST_LABEL_DICT[datatype[1]]["test"])

    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            _run_estimator_method(estimator, method, datatype, "test")

        assert deep_equals(X, TEST_DATA_DICT[datatype[0]]["test"]) and deep_equals(
            y, TEST_LABEL_DICT[datatype[1]]["test"]
        ), f"Estimator: {type(estimator)} has side effects on arguments of {method}"

        # dict_after = dictionary of estimator after predict and fit
        dict_after = estimator.__dict__
        is_equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
        assert is_equal, (
            f"Estimator: {type(estimator).__name__} changes __dict__ "
            f"during {method}, "
            f"reason/location of discrepancy (x=after, y=before): {msg}"
        )


def check_fit_updates_state(estimator, datatype):
    """Check fit/update state change.

    1. Check estimator_instance calls base class constructor
    2. Check is_fitted attribute is set correctly to False before fit, at init
        This is testing base class functionality, but its fast
    3. Check fit returns self
    4. Check is_fitted attribute is updated correctly to True after calling fit
    5. Check estimator hyper parameters are not changed in fit
    """
    # Check that fit updates the is-fitted states
    attrs = ["_is_fitted", "is_fitted"]

    estimator = _clone_estimator(estimator)

    msg = (
        f"{type(estimator).__name__}.__init__ should call "
        f"super({type(estimator).__name__}, self).__init__, "
        "but that does not seem to be the case. Please ensure to call the "
        f"parent class's constructor in {type(estimator).__name__}.__init__"
    )
    assert hasattr(estimator, "_is_fitted"), msg

    # Check is_fitted attribute is set correctly to False before fit, at init
    for attr in attrs:
        assert not getattr(
            estimator, attr
        ), f"Estimator: {estimator} does not initiate attribute: {attr} to False"

    # Make a physical copy of the original estimator parameters before fitting.
    original_params = deepcopy(estimator.get_params())

    fitted_estimator = _run_estimator_method(estimator, "fit", datatype, "train")

    # Check fit returns self
    assert (
        fitted_estimator is estimator
    ), f"Estimator: {estimator} does not return self when calling fit"

    # Check is_fitted attribute is updated correctly to True after calling fit
    for attr in attrs:
        assert getattr(
            fitted_estimator, attr
        ), f"Estimator: {estimator} does not update attribute: {attr} during fit"

    # Compare the state of the model parameters with the original parameters
    new_params = fitted_estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (estimator.__class__.__name__, param_name, original_value, new_value)
        )


def check_raises_not_fitted_error(estimator, datatype):
    """Check exception raised for non-fit method calls to unfitted estimators.

    Tries to run all methods in NON_STATE_CHANGING_METHODS with valid scenario,
    but before fit has been called on the estimator.

    This should raise a NotFittedError if correctly caught,
    normally by a self.check_is_fitted() call in the method's boilerplate.

    Raises
    ------
    Exception if NotFittedError is not raised by non-state changing method
    """
    # call methods without prior fitting and check that they raise NotFittedError
    for method in NON_STATE_CHANGING_METHODS:
        if hasattr(estimator, method):
            with pytest.raises(NotFittedError, match=r"has not been fitted"):
                _run_estimator_method(estimator, method, datatype, "test")


def test_persistence_via_pickle(estimator, datatype):
    """Check that we can pickle all estimators."""
    estimator = _clone_estimator(estimator, random_state=0)
    _run_estimator_method(estimator, "fit", datatype, "train")

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method):
            output = _run_estimator_method(estimator, method, datatype, "test")
            results.append(output)

    # Serialize and deserialize
    serialized_estimator = pickle.dumps(estimator)
    estimator = pickle.loads(serialized_estimator)

    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method):
            output = _run_estimator_method(estimator, method, datatype, "test")

            _assert_array_almost_equal(
                output,
                results[i],
                err_msg=f"Running {method} after fit twice with test "
                f"parameters gives different results.",
            )

            i += 1


def check_fit_deterministic(estimator, datatype):
    """Test that fit is deterministic.

    Check that calling fit twice is equivalent to calling it once.
    """
    estimator = _clone_estimator(estimator, random_state=0)
    _run_estimator_method(estimator, "fit", datatype, "train")

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method):
            output = _run_estimator_method(estimator, method, datatype, "test")
            results.append(output)

    # run fit and other methods a second time
    _run_estimator_method(estimator, "fit", datatype, "train")

    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method):
            output = _run_estimator_method(estimator, method, datatype, "test")

            _assert_array_almost_equal(
                output,
                results[i],
                err_msg=f"Running {method} after fit twice with test "
                f"parameters gives different results.",
            )

            i += 1


# def check_multiprocessing_idempotent(estimator):
#     """Test that single and multi-process run results are identical.
#
#     Check that running an estimator on a single process is no different to running
#     it on multiple processes. We also check that we can set n_jobs=-1 to make use
#     of all CPUs. The test is not really necessary though, as we rely on joblib for
#     parallelization and can trust that it works as expected.
#     """
#     method_nsc = method_nsc_arraylike
#     params = estimator_instance.get_params()
#
#     if "n_jobs" in params:
#         # run on a single process
#         # -----------------------
#         estimator = deepcopy(estimator_instance)
#         estimator.set_params(n_jobs=1)
#         set_random_state(estimator)
#         result_single_process = scenario.run(
#             estimator, method_sequence=["fit", method_nsc]
#         )
#
#         # run on multiple processes
#         # -------------------------
#         estimator = deepcopy(estimator_instance)
#         estimator.set_params(n_jobs=-1)
#         set_random_state(estimator)
#         result_multiple_process = scenario.run(
#             estimator, method_sequence=["fit", method_nsc]
#         )
#         _assert_array_equal(
#             result_single_process,
#             result_multiple_process,
#             err_msg="Results are not equal for n_jobs=1 and n_jobs=-1",
#         )
