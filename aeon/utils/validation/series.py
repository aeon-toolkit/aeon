"""Functions for checking input data."""

__all__ = [
    "check_series",
    "check_time_index",
    "check_equal_time_index",
    "check_consistent_index_type",
    "is_hierarchical",
    "is_single_series",
]
__maintainer__ = ["TonyBagnall"]

from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

# We currently support the following types for input data and time index types.
VALID_DATA_TYPES = (pd.DataFrame, pd.Series, np.ndarray)
VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)
RELATIVE_INDEX_TYPES = (pd.RangeIndex, pd.TimedeltaIndex)
ABSOLUTE_INDEX_TYPES = (pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)
assert set(RELATIVE_INDEX_TYPES).issubset(VALID_INDEX_TYPES)
assert set(ABSOLUTE_INDEX_TYPES).issubset(VALID_INDEX_TYPES)


def is_single_series(y):
    """Check if input is a single time series.

    Minimal checks that do not check the index characteristics. To check index and
    throw an error if not correct, use `check_series` instead.

    Parameters
    ----------
    y : Any object

    Returns
    -------
    bool
        True if y is one of VALID_DATA_TYPES a valid shape with unique columns.
    """
    if isinstance(y, pd.Series):
        return True
    if isinstance(y, pd.DataFrame):
        if "object" in y.dtypes.values:
            return False
        if y.index.nlevels > 1:
            return False
        return True
    if isinstance(y, np.ndarray):
        if y.ndim > 2:
            return False
        return True
    return False


def is_hierarchical(y):
    """Check to see if y is in a hierarchical dataframe.

     Hierarchical is defined as a pd.DataFrame having 3 or more indices.

    Parameters
    ----------
    y : Any object

    Returns
    -------
    bool
        True if y is a pd.DataFrame with three or more indices.
    """
    if isinstance(y, pd.DataFrame):
        if y.index.nlevels >= 3:
            return True
    return False


def _check_pd_dataframe(y):
    # check that columns are unique
    if not y.columns.is_unique:
        raise ValueError(
            f"Series in a pd.DataFrame must have unique column indices " f"{y.columns}"
        )
    # check whether the time index is of valid type
    if not is_in_valid_index_types(y.index):
        raise ValueError(
            f"{type(y.index)} is not supported for series, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
    # check that no dtype is object
    if "object" in y.dtypes.values:
        raise ValueError("y should not have column of 'object' dtype")
    # Check time index is ordered in time
    if not y.index.is_monotonic_increasing:
        raise ValueError(
            f"The (time) index of a series must be sorted monotonically increasing, "
            f"but found: {y.index}"
        )


def is_univariate_series(y):
    """Check if series is univariate.

    Parameters
    ----------
    y : series
        Time series data.

    Returns
    -------
    bool
        True if series is pd.Series, single column pd.DataFrame or np.ndarray with 1
        dimension, False otherwise.
    """
    if isinstance(y, pd.Series):
        return True
    if isinstance(y, pd.DataFrame):
        nvars = y.shape[1]
        if nvars > 1:
            return False
        return True
    if isinstance(y, np.ndarray):
        if y.ndim > 1 and y.shape[1] > 1:
            return False
        return True
    return False


def get_index_for_series(obj, cutoff=0):
    """Get pandas index for a Series object.

    Returns index even for numpy array, in that case a RangeIndex.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : data structure
        must be of one of pd.Series, pd.DataFrame, np.ndarray
    cutoff : int, or pd.datetime, optional, default=0
        current cutoff, used to offset index if obj is np.ndarray

    Returns
    -------
    index : pandas.Index, index for obj
    """
    if hasattr(obj, "index"):
        return obj.index
    # now we know the object must be an np.ndarray
    return pd.RangeIndex(cutoff, cutoff + obj.shape[0])


def is_integer_index(x) -> bool:
    """Check that the input is an integer pd.Index."""
    return isinstance(x, pd.Index) and is_integer_dtype(x)


def is_in_valid_index_types(x) -> bool:
    """Check that the input type belongs to the valid index types."""
    return isinstance(x, VALID_INDEX_TYPES) or is_integer_index(x)


def is_in_valid_relative_index_types(x) -> bool:
    return isinstance(x, RELATIVE_INDEX_TYPES) or is_integer_index(x)


def is_in_valid_absolute_index_types(x) -> bool:
    return isinstance(x, ABSOLUTE_INDEX_TYPES) or is_integer_index(x)


def check_is_univariate(y, var_name="input"):
    """Check if series is univariate."""
    if isinstance(y, pd.DataFrame):
        nvars = y.shape[1]
        if nvars > 1:
            raise ValueError(
                f"{var_name} must be univariate, but found {nvars} variables."
            )
    if isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(
            f"{var_name} must be univariate, but found np.ndarray with more than "
            "one column"
        )


def _check_is_multivariate(Z, var_name="input"):
    """Check if series is multivariate.

    Warning: this function assumes ndarrays are in (n_timepoints, n_channels) shape. Do
    not use with collections of time series.
    """
    if isinstance(Z, pd.Series):
        raise ValueError(f"{var_name} must have 2 or more variables, but found 1.")
    if isinstance(Z, pd.DataFrame):
        nvars = Z.shape[1]
        if nvars < 2:
            raise ValueError(
                f"{var_name} must have 2 or more variables, but found {nvars}."
            )
    if isinstance(Z, np.ndarray):
        if Z.ndim == 1 or (Z.ndim == 2 and Z.shape[1] == 1):
            raise ValueError(f"{var_name} must have 2 or more variables, but found 1.")


def check_series(
    Z,
    enforce_univariate=False,
    enforce_multivariate=False,
    allow_empty=False,
    allow_numpy=True,
    allow_None=True,
    enforce_index_type=None,
    allow_index_names=False,
    var_name="input",
):
    """Validate input data to be a valid type for Series.

    Parameters
    ----------
    Z : pd.Series, pd.DataFrame, np.ndarray, or None
        Univariate or multivariate time series.
    enforce_univariate : bool, default = False
        If True, multivariate Z will raise an error.
    enforce_multivariate: bool, default = False
        If True, univariate Z will raise an error.
    allow_empty : bool, default = False
        whether a container with zero samples is allowed
    allow_numpy : bool, default = True
        whether no error is raised if Z is in a valid numpy.ndarray format
    allow_None : bool, default = True
        whether no error is raised if Z is None
    enforce_index_type : type, default = None
        type of time index
    allow_index_names : bool, default = False
        If False, names of Z.index will be set to None
    var_name : str, default = "input" - variable name printed in error messages

    Returns
    -------
    Z : pd.Series, pd.DataFrame, np.ndarray, or None
        Validated time series - a reference to the input Z

    Raises
    ------
    TypeError - if Z is not in a valid type for Series
    if enforce_univariate is True:
        ValueError if Z has 2 or more columns
    if enforce_multivariate is True:
        ValueError if Z has 1 column
    if allow_numpy is false:
        TypeError - if Z is of type np.ndarray
    if allow_empty is false:
        ValueError - if Z has length 0
    if allow_None is false:
        ValueError - if Z is None
    if enforce_index_type is not None and Z is pandas type:
        ValueError - if Z has index type other than enforce_index_type
    """
    if Z is None:
        if allow_None:
            return Z
        else:
            raise ValueError(var_name + " cannot be None")

    # Check if pandas series or numpy array
    if not allow_numpy:
        valid_data_types = tuple(
            filter(lambda x: x is not np.ndarray, VALID_DATA_TYPES)
        )
    else:
        valid_data_types = VALID_DATA_TYPES

    if not isinstance(Z, valid_data_types):
        raise TypeError(
            f"{var_name} must be a one of {valid_data_types}, but found type: {type(Z)}"
        )

    if enforce_univariate and enforce_multivariate:
        raise ValueError(
            "`enforce_univariate` and `enforce_multivariate` cannot both be set to "
            "True."
        )

    if enforce_univariate:
        check_is_univariate(Z, var_name=var_name)

    if enforce_multivariate:
        _check_is_multivariate(Z, var_name=var_name)

    # check time index if input data is not an NumPy ndarray
    if not isinstance(Z, np.ndarray):
        check_time_index(
            Z.index,
            allow_empty=allow_empty,
            enforce_index_type=enforce_index_type,
            var_name=var_name,
        )

    if not allow_index_names and not isinstance(Z, np.ndarray):
        Z.index.names = [None for name in Z.index.names]

    return Z


def check_time_index(
    index: Union[pd.Index, np.array],
    allow_empty: bool = False,
    enforce_index_type: bool = None,
    var_name: str = "input",
) -> pd.Index:
    """Check time index.

    Parameters
    ----------
    index : pd.Index or np.array
        Time index
    allow_empty : bool, optional (default=False)
        If False, empty `index` raises an error.
    enforce_index_type : type, optional (default=None)
        type of time index
    var_name : str, default = "input" - variable name printed in error messages

    Returns
    -------
    time_index : pd.Index
        Validated time index - a reference to the input index
    """
    if isinstance(index, np.ndarray):
        index = pd.Index(index)

    # We here check for type equality because isinstance does not
    # work reliably because index types inherit from each other.
    if not is_in_valid_index_types(index):
        raise NotImplementedError(
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )

    if enforce_index_type and type(index) is not enforce_index_type:
        raise NotImplementedError(
            f"{type(index)} is not supported for {var_name}, use "
            f"type: {enforce_index_type} or integer pd.Index instead."
        )

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
        raise ValueError(
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )

    # Check that index is not empty
    if not allow_empty and len(index) < 1:
        raise ValueError(
            f"{var_name} must contain at least some values, but found none."
        )

    return index


def check_equal_time_index(*ys, mode="equal"):
    """Check that time series have the same (time) indices.

    Parameters
    ----------
    *ys : tuple of aeon compatible time series data containers
        must be pd.Series, pd.DataFrame or 1/2D np.ndarray, or None
        can be Series, Panel, Hierarchical, but must be pandas or numpy
        note: this assumption is not checked by the function itself
    mode : str, "equal" or "contained", optional, default = "equal"
        if "equal" will check for all indices being exactly equal
        if "contained", will check whether all indices are subset of ys[0].index

    Raises
    ------
    ValueError
        if mode = "equal", raised if there are at least two non-None entries of ys
            of which pandas indices are not the same
        if mode = "contained, raised if there is at least one non-None ys[i]
            such that ys[i].index is not contained in ys[o].index
        np.ndarray are considered having (pandas) integer range index on axis 0
    """
    y_not_None = [y for y in ys if y is not None]

    # if there is no or just one element, there is nothing to compare
    if len(y_not_None) < 2:
        return None

    # only validate indices if data is passed as pd.Series
    first_index = get_index_for_series(y_not_None[0])

    for i, y in enumerate(y_not_None[1:]):
        y_index = get_index_for_series(y)

        if mode == "equal":
            failure_cond = not first_index.equals(y_index)
            msg = (
                f"(time) indices are not the same, series 0 and {i} "
                f"differ in the following: {first_index.symmetric_difference(y_index)}."
            )
        elif mode == "contains":
            failure_cond = not y_index.isin(first_index).all()
            msg = (
                f"(time) indices of series {i} are not contained in index of series 0,"
                f" extra indices are: {y_index.difference(first_index)}"
            )
        else:
            raise ValueError('mode must be "equal" or "contains"')

        if failure_cond:
            raise ValueError(msg)


def check_consistent_index_type(a, b):
    """Check that two indices have consistent types.

    Parameters
    ----------
    a : pd.Index
        Index being checked for consistency
    b : pd.Index
        Index being checked for consistency

    Raises
    ------
    TypeError
        If index types are inconsistent
    """
    msg = (
        "Found series with inconsistent index types, please make sure all "
        "series have the same index type."
    )

    if is_integer_index(a):
        if not is_integer_index(b):
            raise TypeError(msg)

    else:
        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        if not type(a) is type(b):  # noqa
            raise TypeError(msg)


def _common_checks(y: pd.DataFrame):
    if not isinstance(y, pd.DataFrame):
        return False
    # check that column indices are unique
    if not len(set(y.columns)) == len(y.columns):
        return False
    # check that all cols are numeric
    if not np.all([is_numeric_dtype(y[c]) for c in y.columns]):
        return False
    # Check time index is ordered in time
    if not y.index.is_monotonic_increasing:
        return False
    return True


def is_pred_interval_proba(y):
    """Check if the input is a dataframe of probas."""
    # we now know obj is a pd.DataFrame
    if not _common_checks(y):
        return False
    # check column multiindex
    colidx = y.columns
    if not isinstance(colidx, pd.MultiIndex) or not colidx.nlevels == 3:
        return False
    coverages = colidx.get_level_values(1)
    if not is_numeric_dtype(coverages):
        return False
    if not (coverages <= 1).all() or not (coverages >= 0).all():
        return False
    upper_lower = colidx.get_level_values(2)
    if not upper_lower.isin(["upper", "lower"]).all():
        return False
    return True


def is_pred_quantiles_proba(y):
    """Check if the input is a dataframe of quantiles."""
    # check if the input is a dataframe
    if not _common_checks(y):
        return False
    # check column multiindex
    colidx = y.columns
    if not isinstance(colidx, pd.MultiIndex) or not colidx.nlevels == 2:
        return False
    alphas = colidx.get_level_values(1)
    if not is_numeric_dtype(alphas):
        return False
    if not (alphas <= 1).all() or not (alphas >= 0).all():
        return False
    return True


def _is_in_valid_multiindex_types(x) -> bool:
    """Check that the input type belongs to the valid multiindex types."""
    return isinstance(x, (pd.RangeIndex, pd.Index)) or is_integer_index(x)


def is_pdmultiindex_hierarchical(y):
    """Check if the input is a pd.DataFrame with MultiIndex.

    Parameters
    ----------
    y : pd.DataFrame
        Input data to be checked.

    Returns
    -------
    bool
        True if y is pd multindex hierarchical.

    """
    if not isinstance(y, pd.DataFrame) or not isinstance(y.index, pd.MultiIndex):
        return False
    if not y.columns.is_unique:
        return False
    # check that there are precisely two index levels
    if y.index.nlevels < 3:
        return False
    # check that no dtype is object
    if "object" in y.dtypes.values:
        return False

    # check whether the time index is of valid type
    if not is_in_valid_index_types(y.index.get_level_values(-1)):
        return False
    time_obj = y.reset_index(-1).drop(y.columns, axis=1)
    time_grp = time_obj.groupby(level=0, group_keys=True, as_index=True)
    inst_inds = time_obj.index.unique()

    # check instance index being integer or range index
    if not _is_in_valid_multiindex_types(inst_inds):
        return False

    if pd.__version__ < "1.5.0":
        # Earlier versions of pandas are very slow for this type of operation.
        montonic_list = [y.loc[i].index.is_monotonic for i in inst_inds]
        time_is_monotonic = len([i for i in montonic_list if i is False]) == 0
    else:
        timedelta_by_grp = (
            time_grp.diff().groupby(level=0, group_keys=True, as_index=True).nunique()
        )
        timedelta_unique = timedelta_by_grp.iloc[:, 0].unique()
        time_is_monotonic = all(timedelta_unique >= 0)
    if not time_is_monotonic:
        return False
    return True
