"""Utility function for estimator testing."""

__maintainer__ = ["MatthewMiddlehurst"]

import inspect
from inspect import isclass

from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.testing.testing_data import FULL_TEST_DATA_DICT


def _run_estimator_method(estimator, method_name, datatype, split):
    method = getattr(estimator, method_name)
    args = inspect.getfullargspec(method)[0]
    try:
        if "X" in args and "length" in args:  # SeriesSearch
            value = method(
                X=FULL_TEST_DATA_DICT[datatype][split][0],
                length=3,
            )
        elif "X" in args and "y" in args:
            value = method(
                X=FULL_TEST_DATA_DICT[datatype][split][0],
                y=FULL_TEST_DATA_DICT[datatype][split][1],
            )
        elif "X" in args:
            value = method(X=FULL_TEST_DATA_DICT[datatype][split][0])
        else:
            value = method()

        # Similarity search return tuple as (distances, indexes)
        if isinstance(estimator, BaseSimilaritySearch):
            if isinstance(value, tuple):
                return value[0]
            else:
                return value
        else:
            return value
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
            tag_name=tag_name, raise_error=raise_error, tag_value_default=default
        )
    else:
        return estimator.get_tag(
            tag_name=tag_name, raise_error=raise_error, tag_value_default=default
        )
