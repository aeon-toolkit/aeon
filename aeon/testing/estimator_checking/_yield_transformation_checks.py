"""Tests for all transformers."""

import sys
from functools import partial

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.base._base_series import VALID_SERIES_INNER_TYPES
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing.expected_results.expected_transform_outputs import (
    basic_motions_result,
    unit_test_result,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.deep_equals import deep_equals
from aeon.testing.utils.estimator_checks import _run_estimator_method
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector
from aeon.transformations.series import BaseSeriesTransformer
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES


def _yield_transformation_checks(estimator_class, estimator_instances, datatypes):
    """Yield all transformation checks for an aeon transformer."""
    # only class required
    if sys.platform == "linux":  # We cannot guarantee same results on ARM macOS
        # Compare against results for both UnitTest and BasicMotions if available
        yield partial(
            check_transformer_against_expected_results,
            estimator_class=estimator_class,
            data_name="UnitTest",
            data_loader=load_unit_test,
            results_dict=unit_test_result,
            resample_seed=0,
        )
        yield partial(
            check_transformer_against_expected_results,
            estimator_class=estimator_class,
            data_name="BasicMotions",
            data_loader=load_basic_motions,
            results_dict=basic_motions_result,
            resample_seed=4,
        )
    yield partial(check_transformer_overrides_and_tags, estimator_class=estimator_class)

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_transformer_output, estimator=estimator, datatype=datatype
            )

            if isinstance(estimator, BaseChannelSelector):
                yield partial(
                    check_channel_selectors,
                    estimator=estimator,
                    datatype=datatype,
                )

            if estimator is not None and estimator.get_tag(
                "capability:inverse_transform"
            ):
                yield partial(
                    check_transform_inverse_transform_equivalent,
                    estimator=estimator,
                    datatype=datatype,
                )


def check_transformer_against_expected_results(
    estimator_class, data_name, data_loader, results_dict, resample_seed
):
    """Test transformer against stored results."""
    # retrieve expected transform output, and skip test if not available
    if estimator_class.__name__ in results_dict.keys():
        expected_results = results_dict[estimator_class.__name__]
    else:
        # skip test if no expected results are registered
        return f"No stored results for {estimator_class.__name__} on {data_name}"

    # we only use the first estimator instance for testing
    estimator_instance = estimator_class._create_test_instance(
        parameter_set="results_comparison"
    )
    # set random seed if possible
    set_random_state(estimator_instance, 0)

    # load test data
    X_train, y_train = data_loader(split="train")
    indices = np.random.RandomState(resample_seed).choice(
        len(y_train), 5, replace=False
    )

    # fit transformer and transform
    results = np.nan_to_num(
        estimator_instance.fit_transform(X_train[indices], y_train[indices]),
        False,
        0,
        0,
        0,
    )

    # assert results are the same
    assert_array_almost_equal(
        results,
        expected_results,
        decimal=2,
        err_msg=f"Failed to reproduce results for {estimator_class.__name__} "
        f"on {data_name}",
    )


def check_transformer_overrides_and_tags(estimator_class):
    """Test does not override final methods."""
    final_methods = [
        "fit",
        "transform",
        "fit_transform",
    ]
    for method in final_methods:
        if method in estimator_class.__dict__:
            raise ValueError(
                f"Transformer {estimator_class} overrides the method {method}. "
                f"Override _{method} instead."
            )

    dtypes = (
        VALID_SERIES_INNER_TYPES
        if issubclass(estimator_class, BaseSeriesTransformer)
        else COLLECTIONS_DATA_TYPES
    )

    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in dtypes
    else:  # must be a list
        assert all([t in dtypes for t in X_inner_type])

    # one of X_inner_types must be capable of storing unequal length
    if estimator_class.get_class_tag(
        "capability:unequal_length", raise_error=False, tag_value_default=False
    ):
        valid_unequal_types = ["np-list", "df-list", "pd-multiindex"]
        if isinstance(X_inner_type, str):
            assert X_inner_type in valid_unequal_types
        else:  # must be a list
            assert any([t in valid_unequal_types for t in X_inner_type])

    if estimator_class.get_class_tag("capability:inverse_transform"):
        assert "_inverse_transform" in estimator_class.__dict__
    else:
        assert "_inverse_transform" not in estimator_class.__dict__


def check_transformer_output(estimator, datatype):
    """Test transformer outputs."""
    estimator = _clone_estimator(estimator)
    set_random_state(estimator, 0)

    # run fit and predict
    _run_estimator_method(estimator, "fit", datatype, "train")
    Xt = _run_estimator_method(estimator, "transform", datatype, "train")

    if "_fit_transform" in estimator.__class__.__dict__:
        Xt2 = _run_estimator_method(estimator, "fit_transform", datatype, "train")
        assert deep_equals(Xt, Xt2, ignore_index=True)


def check_channel_selectors(estimator, datatype):
    """Test channel selectors have fit and select at least one channel."""
    estimator = _clone_estimator(estimator)

    assert not estimator.get_tag("fit_is_empty")

    Xt = _run_estimator_method(estimator, "fit_transform", datatype, "train")

    assert hasattr(estimator, "channels_selected_")
    assert isinstance(estimator.channels_selected_, (list, np.ndarray))
    assert len(estimator.channels_selected_) > 0
    assert isinstance(Xt, np.ndarray)
    assert Xt.ndim == 3


def check_transform_inverse_transform_equivalent(estimator, datatype):
    """Test that inverse_transform is inverse to transform."""
    estimator = _clone_estimator(estimator)

    X = FULL_TEST_DATA_DICT[datatype]["train"][0]
    Xt = _run_estimator_method(estimator, "fit_transform", datatype, "train")
    Xit = estimator.inverse_transform(Xt)

    if isinstance(X, (np.ndarray, pd.DataFrame)):
        X = X.squeeze()
    if isinstance(Xit, (np.ndarray, pd.DataFrame)):
        Xit = Xit.squeeze()

    assert deep_equals(X, Xit, ignore_index=True)
