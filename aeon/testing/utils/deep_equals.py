"""Testing utility to compare equality in value for nested objects."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["deep_equals"]

from inspect import isclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def deep_equals(x, y, ignore_index=False, return_msg=False):
    """Test two objects for equality in value.

    Intended for:
        pd.Series, pd.DataFrame, np.ndarray, lists, tuples, or dicts.
        Will recursively compare nested objects.

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
        First item to compare.
    y : object
        Second item to compare.
    ignore_index : bool, default=False
        If True, will ignore the index of pd.Series and pd.DataFrame.
    return_msg : bool, default=False
        Whether to return an informative message about what is not equal.

    Returns
    -------
    is_equal: bool
        True if x and y are equal in value, x and y do not need to be equal in
        reference.
    msg : str
        Only returned if return_msg is True.
        Indication of what is the reason for not being equal
    """
    eq, msg = _deep_equals(x, y, 0, ignore_index)
    return eq if not return_msg else (eq, msg)


def _deep_equals(x, y, depth, ignore_index):
    if x is y:
        return True, ""
    if type(x) is not type(y):
        return False, f"x.type ({type(x)}) != y.type ({type(y)}), depth={depth}"

    if isinstance(x, pd.Series):
        return _series_equals(x, y, depth, ignore_index)
    elif isinstance(x, pd.DataFrame):
        return _dataframe_equals(x, y, depth, ignore_index)
    elif isinstance(x, np.ndarray):
        return _numpy_equals(x, y, depth, ignore_index)
    elif isinstance(x, (list, tuple)):
        return _list_equals(x, y, depth, ignore_index)
    elif isinstance(x, dict):
        return _dict_equals(x, y, depth, ignore_index)
    elif isinstance(x, csr_matrix):
        return _csrmatrix_equals(x, y, depth)
    # non-iterable types
    elif isclass(x):
        eq = x == y
        msg = "" if eq else f"x ({x.__name__}) != y ({y.__name__}), depth={depth}"
        return eq, msg
    elif np.isnan(x):
        eq = np.isnan(y)
        msg = "" if eq else f"x ({x}) != y ({y}), depth={depth}"
        return eq, msg
    elif isinstance(x == y, (bool, np.bool_)):
        eq = x == y
        msg = "" if eq else f"x ({x}) != y ({y}), depth={depth}"
        return eq, msg
    # unknown type
    else:
        raise ValueError(f"Unknown type: {type(x)}, depth={depth}")


def _series_equals(x, y, depth, ignore_index):
    if x.dtype != y.dtype:
        return False, f"x.dtype ({x.dtype}) != y.dtype ({y.dtype}), depth={depth}"
    if x.shape != y.shape:
        return False, f"x.shape ({x.shape}) != y.shape ({y.shape}), depth={depth}"

    # if columns are object, recurse over entries and index
    if x.dtype == "object":
        index_equal = ignore_index or x.index.equals(y.index)
        values_equal, values_msg = _deep_equals(
            list(x.values), list(y.values), depth, ignore_index
        )

        if not values_equal:
            msg = values_msg
        elif not index_equal:
            msg = f".index, x.index: {x.index}, y.index: {y.index}, depth={depth}"
        else:
            msg = ""

        return index_equal and values_equal, msg
    else:
        eq = x.equals(y)
        msg = "" if eq else f"x ({x}) != y ({y}), depth={depth}"
        return eq, msg


def _dataframe_equals(x, y, depth, ignore_index):
    if not x.columns.equals(y.columns):
        return (
            False,
            f"x.columns ({x.columns}) != y.columns ({y.columns}), depth={depth}",
        )
    if x.shape != y.shape:
        return False, f"x.shape ({x.shape}) != y.shape ({y.shape}), depth={depth}"

    # if columns are equal and at least one is object, recurse over Series
    if sum(x.dtypes == "object") > 0:
        for i, c in enumerate(x.columns):
            eq, msg = _deep_equals(x[c], y[c], depth + 1, ignore_index)

            if not eq:
                return False, msg + f", idx={i}"
        return True, ""
    else:
        eq = (
            np.allclose(x.values, y.values, equal_nan=True)
            if ignore_index
            else x.equals(y)
        )
        msg = "" if eq else f"x ({x}) != y ({y}), depth={depth}"
        return eq, msg


def _numpy_equals(x, y, depth, ignore_index):
    if x.dtype != y.dtype:
        return False, f"x.dtype ({x.dtype}) != y.dtype ({y.dtype}), depth={depth}"
    if x.shape != y.shape:
        return False, f"x.shape ({x.shape}) != y.shape ({y.shape}), depth={depth}"

    if x.dtype == "object":
        for i in range(len(x)):
            eq, msg = _deep_equals(x[i], y[i], depth + 1, ignore_index)

            if not eq:
                return False, msg + f", idx={i}"
    else:
        eq = np.allclose(x, y, equal_nan=True)
        msg = "" if eq else f"x ({x}) != y ({y}), depth={depth}"
        return eq, msg
    return True, ""


def _csrmatrix_equals(x, y, depth):
    if not np.allclose(x.toarray(), y.toarray(), equal_nan=True):
        return False, f"x ({x}) !=  y ({y}), depth={depth}"
    return True, ""


def _list_equals(x, y, depth, ignore_index):
    if len(x) != len(y):
        return False, f"x.len ({len(x)}) != y.len ({len(y)}), depth={depth}"

    for i in range(len(x)):
        eq, msg = _deep_equals(x[i], y[i], depth + 1, ignore_index)

        if not eq:
            return False, msg + f", idx={i}"
    return True, ""


def _dict_equals(x, y, depth, ignore_index):
    xkeys = set(x.keys())
    ykeys = set(y.keys())
    if xkeys != ykeys:
        xmy = xkeys.difference(ykeys)
        ymx = ykeys.difference(xkeys)

        msg = "x.keys != y.keys"
        if len(xmy) > 0:
            msg += f", x.keys-y.keys = {xmy}"
        if len(ymx) > 0:
            msg += f", y.keys-x.keys = {ymx}"

        return False, msg + f", depth={depth}"

    # we now know that xkeys == ykeys
    for i, key in enumerate(xkeys):
        eq, msg = _deep_equals(x[key], y[key], depth + 1, ignore_index)

        if not eq:
            return False, msg + f", idx={i}"
    return True, ""
