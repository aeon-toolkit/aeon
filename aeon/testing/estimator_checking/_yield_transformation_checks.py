"""Tests for all transformers."""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.deep_equals import deep_equals
from aeon.testing.utils.estimator_checks import _run_estimator_method
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector
from aeon.transformations.series import BaseSeriesTransformer
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES, VALID_SERIES_INNER_TYPES


def _yield_transformation_checks(estimator_class, estimator_instances, datatypes):
    """Yield all transformation checks for an aeon transformer."""
    # only class required
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

        Xt3 = _run_estimator_method(estimator, "transform", datatype, "train")
        assert deep_equals(Xt, Xt3, ignore_index=True)


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
    # If input is equal length, inverse should be equal length
    if isinstance(X, np.ndarray) and isinstance(Xit, list):
        Xit = np.array(Xit)
        Xit = Xit.squeeze()
    eq, msg = deep_equals(X, Xit, ignore_index=True, return_msg=True)
    assert eq, msg
