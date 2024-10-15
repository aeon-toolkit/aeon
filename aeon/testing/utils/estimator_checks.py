"""Utility function for estimator testing."""

__maintainer__ = []

import inspect
from inspect import isclass, signature

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from aeon.base import BaseAeonEstimator
from aeon.clustering.base import BaseClusterer
from aeon.regression.base import BaseRegressor
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.transformations.base import BaseTransformer
from aeon.utils.validation import is_nested_univ_dataframe


def _run_estimator_method(estimator, method_name, datatype, split):
    method = getattr(estimator, method_name)
    args = inspect.getfullargspec(method)[0]
    try:
        if "X" in args and "y" in args:
            return method(
                X=FULL_TEST_DATA_DICT[datatype][split][0],
                y=FULL_TEST_DATA_DICT[datatype][split][1],
            )
        elif "X" in args:
            return method(X=FULL_TEST_DATA_DICT[datatype][split][0])
        else:
            return method()
    # generic message for ModuleNotFoundError which are assumed to be related to
    # soft dependencies
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Estimator {estimator.__name__} raises a ModuleNotFoundError "
            f"on {method.__name__}. Any required soft dependencies should "
            f'be added to the "python_dependencies" tag, and python version bounds '
            f'should be added to the "python_version" tag.'
        ) from e


def _get_tag(estimator, tag_name, default=None, raise_error=False):
    if estimator is None:
        return None
    elif isclass(estimator):
        return estimator.get_class_tag(
            tag_name=tag_name, tag_value_default=default, raise_error=raise_error
        )
    else:
        return estimator.get_tag(
            tag_name=tag_name, tag_value_default=default, raise_error=raise_error
        )


def _list_required_methods(estimator):
    """Return list of required method names (beyond BaseAeonEstimator ones)."""
    # all BaseAeonEstimator children must implement these
    MUST_HAVE_FOR_OBJECTS = ["set_params", "get_params"]

    # all BaseAeonEstimator children must implement these
    MUST_HAVE_FOR_ESTIMATORS = [
        "fit",
        "check_is_fitted",
        "is_fitted",  # read-only property
    ]
    # prediction/forecasting base classes that must have predict
    BASE_CLASSES_THAT_MUST_HAVE_PREDICT = (
        BaseClusterer,
        BaseRegressor,
    )
    # transformation base classes that must have transform
    BASE_CLASSES_THAT_MUST_HAVE_TRANSFORM = (BaseTransformer,)

    required_methods = []

    if isinstance(estimator, BaseAeonEstimator):
        required_methods += MUST_HAVE_FOR_OBJECTS

    if isinstance(estimator, BaseAeonEstimator):
        required_methods += MUST_HAVE_FOR_ESTIMATORS

    if isinstance(estimator, BASE_CLASSES_THAT_MUST_HAVE_PREDICT):
        required_methods += ["predict"]

    if isinstance(estimator, BASE_CLASSES_THAT_MUST_HAVE_TRANSFORM):
        required_methods += ["transform"]

    return required_methods


def _compare_nested_frame(func, x, y, **kwargs):
    """Compare two nested pd.DataFrames.

    Parameters
    ----------
    func : function
        Function from np.testing for comparing arrays.
    x : pd.DataFrame
    y : pd.DataFrame
    kwargs : dict
        Keyword argument for function

    Raises
    ------
    AssertionError
        If x and y are not equal
    """
    # We iterate over columns and rows to make cell-wise comparisons.
    # Tabularizing the data first would simplify this, but does not
    # work for unequal length data.

    # In rare cases, x and y may be empty (e.g. TSFreshRelevantFeatureExtractor) and
    # we cannot compare individual cells, so we simply check if everything else is
    # equal here.
    assert isinstance(x, pd.DataFrame)
    if x.empty:
        assert_frame_equal(x, y)

    elif is_nested_univ_dataframe(x):
        # Check if both inputs have the same shape
        if not x.shape == y.shape:
            raise ValueError("Found inputs with different shapes")

        # Iterate over columns
        n_columns = x.shape[1]
        for i in range(n_columns):
            xc = x.iloc[:, i].tolist()
            yc = y.iloc[:, i].tolist()

            # Iterate over rows, checking if individual cells are equal
            for xci, yci in zip(xc, yc):
                func(xci, yci, **kwargs)


def _assert_array_almost_equal(x, y, decimal=6, err_msg=""):
    func = np.testing.assert_array_almost_equal
    if isinstance(x, pd.DataFrame):
        _compare_nested_frame(func, x, y, decimal=decimal, err_msg=err_msg)
    else:
        func(x, y, decimal=decimal, err_msg=err_msg)


def _get_args(function, varargs=False):
    """Get function arguments."""
    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [
        key
        for key, param in params.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    if varargs:
        varargs = [
            param.name
            for param in params.values()
            if param.kind == param.VAR_POSITIONAL
        ]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args
