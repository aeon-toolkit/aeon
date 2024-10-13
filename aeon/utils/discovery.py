"""aeon discovery methods."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["all_estimators"]

import inspect
import warnings
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from pkgutil import walk_packages

from sklearn.base import BaseEstimator

from aeon.base import BaseAeonEstimator
from aeon.utils.base import BASE_CLASS_REGISTER
from aeon.utils.tags import ESTIMATOR_TAGS
from aeon.utils.tags._validate import check_tag_value


def all_estimators(
    type_filter=None,
    exclude_types=None,
    tag_filter=None,
    exclude_tags=None,
    include_sklearn=True,
    return_names=True,
):
    """Get a list of all estimators from aeon.

    This function crawls the module and gets all classes that inherit
    from aeon's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Based on the sklearn utility of the same name.

    Parameters
    ----------
    type_filter: str, list of str, default=None
        Which kind of estimators should be returned.
        if None, no filter is applied and all estimators are returned.
        if str or list of str, string identifiers define types specified in search
                only estimators that are of (at least) one of the types are returned
            possible str values are entries of registry.BASE_CLASS_REGISTER (first col)
                for instance 'classifier', 'regressor', 'transformer'
    exclude_types: str, list of str, default=None
        Names of estimator types to exclude i.e. "collection_transformer" when you are
        looking for "transformer" classes
    tag_filter: dict of (str or list of str), default=None
        For a list of valid tag strings, use the registry.all_tags utility.
        subsets the returned estimators as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"
    exclude_tags: str or list of str, default=None
        Names of tags to fetch and return each estimator's value of.
        For a list of valid tag strings, use the registry.all_tags utility.
        if str or list of str,
            the tag values named in return_tags will be fetched for each
            estimator and will be appended as either columns or tuple entries.
    include_sklearn: bool, default=True
        todo
    return_names: bool, default=True
        if True, estimator class name is included in the all_estimators()
            return in the order: name, estimator class, optional tags, either as
            a tuple or as pandas.DataFrame columns
        if False, estimator class name is removed from the all_estimators()
            return.

    Returns
    -------
    estimators
        list of tuples (estimator name, estimator class) if return_names=True,
        else list of estimator classes

    Examples
    --------
    >>> from aeon.utils.discovery import all_estimators
    >>> # return a complete list of estimators as pd.Dataframe
    >>> all = all_estimators(as_dataframe=True)
    >>> # return all classifiers by filtering for estimator type
    >>> classifiers = all_estimators("classifier")
    >>> # return all classifiers which handle unequal length data by tag filtering
    >>> clf_ul = all_estimators(
    ...     "classifier", tag_filter={"capability:unequal_length":True}
    ... )
    """
    modules_to_ignore = (
        # ignore test modules and base classes
        "base",
        "tests",
        # ignore these submodules
        "benchmarking",
        "datasets",
        "distances",
        "networks",
        "performance_metrics",
        "pipeline",
        "testing",
        "utils",
        "visualisation",
    )
    base_filter = BaseEstimator if include_sklearn else BaseAeonEstimator
    root = str(Path(__file__).parent.parent)  # aeon package root directory
    estimators = []

    # ignore deprecation warnings triggered at import time and from walking packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        for _, module_name, _ in walk_packages(path=[root], prefix="aeon."):
            module_parts = module_name.split(".")
            if (
                any(part in modules_to_ignore for part in module_parts)
                or "._" in module_name
            ):
                continue
            module = import_module(module_name)

            classes = inspect.getmembers(module, inspect.isclass)
            # skip private estimators and those not implemented in aeon
            classes = [
                (name, est_cls)
                for name, est_cls in classes
                if not name.startswith("_")
                and str(est_cls).startswith("<class 'aeon")
                and issubclass(est_cls, base_filter)
            ]

            estimators.extend(classes)

    # drop duplicates
    estimators = set(estimators)
    # drop abstract classes
    estimators = [c for c in estimators if not _is_abstract(c[1])]

    # filter based on wanted/excluded
    if type_filter is not None:
        estimators = _filter_types(type_filter, estimators, "type_filter")
    if exclude_types is not None:
        estimators = set(estimators) - set(
            _filter_types(exclude_types, estimators, "exclude_types")
        )

    # filter based on wanted/excluded tags
    if tag_filter is not None:
        estimators = _filter_tags(tag_filter, estimators, "tag_filter")
    if exclude_tags is not None:
        estimators = set(estimators) - set(
            _filter_tags(exclude_tags, estimators, "exclude_tags")
        )

    # sort for reproducibility, remove names if return_names=False
    estimators = sorted(set(estimators), key=itemgetter(0))
    if not return_names:
        return [est for (name, est) in estimators]
    else:
        return estimators


def _is_abstract(c):
    if not (hasattr(c, "__abstractmethods__")):
        return False
    if not len(c.__abstractmethods__):
        return False
    return True


def _filter_types(types, estimators, name):
    msg = (
        f"Parameter {name} must be None, a string or type, or a list of "
        f"strings or types. Valid string/type values are: "
        f"{BASE_CLASS_REGISTER}. Found: {types}"
    )

    if not isinstance(types, list):
        types = [types]
    else:
        types = list(types)  # copy

    filtered_estimators = []
    for t in types:
        if isinstance(t, str):
            if t not in BASE_CLASS_REGISTER.keys():
                raise ValueError(msg)
            filtered_estimators.extend(
                [
                    est
                    for est in estimators
                    if issubclass(est[1], BASE_CLASS_REGISTER[t])
                ]
            )
        elif isinstance(t, type):
            if t not in BASE_CLASS_REGISTER.values():
                raise ValueError(msg)
            filtered_estimators.extend(
                [est for est in estimators if issubclass(est[1], t)]
            )
        else:
            raise ValueError(msg)

    return filtered_estimators


def _filter_tags(tags, estimators, name):
    if tags is None:
        return estimators

    msg = (
        f"Parameter {name} must be None or a dict of tag/value pairs. "
        f"Valid tags are found in aeon.utils.tags.ESTIMATOR_TAGS. Found: {tags}"
    )

    if not isinstance(tags, dict):
        raise TypeError(msg)

    for key, value in tags.items():
        if key not in ESTIMATOR_TAGS:
            raise ValueError(f"Tag {key} is not a valid tag. {msg}")
        if not check_tag_value(key, value, raise_error=False):
            raise ValueError(f"Value {value} is not a valid value for tag {key}. {msg}")

    filtered_estimators = []
    for est in estimators:
        if not issubclass(est[1], BaseAeonEstimator):
            continue

        cond_sat = True
        for key, value in tags.items():
            est_tag = est[1].get_class_tag(key)
            est_tag = est_tag if isinstance(est_tag, list) else [est_tag]

            if isinstance(value, list):
                in_list = False
                for n in value:
                    if n in est_tag:
                        in_list = True
                cond_sat = cond_sat and in_list
            else:
                cond_sat = cond_sat and value in est_tag

        if cond_sat:
            filtered_estimators.append(est)

    return filtered_estimators
