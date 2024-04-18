"""Index functions of dubious worth."""

import numpy as np
import pandas as pd

from aeon.utils.conversion import convert_collection, convert_series
from aeon.utils.validation import (
    is_collection,
    is_hierarchical,
    is_single_series,
    validate_input,
)


def _get_index(x):
    if hasattr(x, "index"):
        return x.index
    elif isinstance(x, np.ndarray):
        if x.ndim < 3:
            return pd.RangeIndex(x.shape[0])
        else:
            return pd.RangeIndex(x.shape[-1])
    else:
        return pd.RangeIndex(x.shape[-1])


def get_time_index(X):
    """Get index of time series data, helper function.

    Parameters
    ----------
    X : pd.DataFrame, pd.Series, np.ndarray
        in one of the following data container types pd.DataFrame, pd.Series,
        np.ndarray, pd-multiindex, nested_univ, pd_multiindex_hier
        assumes all time series have equal length and equal index set
        will *not* work for list-of-df, pd-wide, pd-long, numpy2D or np-list.

    Returns
    -------
    time_index : pandas.Index
        Index of time series
    """
    # assumes that all samples share the same the time index, only looks at
    # first row
    if isinstance(X, (pd.DataFrame, pd.Series)):
        # pd-multiindex or pd_multiindex_hier
        if isinstance(X.index, pd.MultiIndex):
            index_tuple = tuple(list(X.index[0])[:-1])
            index = X.loc[index_tuple].index
            return index
        # nested_univ
        elif isinstance(X, pd.DataFrame) and isinstance(X.iloc[0, 0], pd.DataFrame):
            return _get_index(X.iloc[0, 0])
        # pd.Series or pd.DataFrame
        else:
            return X.index
    # numpy3D and np.ndarray
    elif isinstance(X, np.ndarray):
        # np.ndarray
        if X.ndim < 3:
            return pd.RangeIndex(X.shape[0])
        # numpy3D
        else:
            return pd.RangeIndex(X.shape[-1])
    elif hasattr(X, "X"):
        return get_time_index(X.X)
    else:
        raise ValueError(
            f"X must be pd.DataFrame, pd.Series, or np.ndarray, but found: {type(X)}"
        )


def get_index_for_series(obj, cutoff=0):
    """Get pandas index for a Series object.

    Returns index even for numpy array, in that case a RangeIndex.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : aeon data container
        must be of one of the following single series data structures:
            pd.Series, pd.DataFrame, np.ndarray
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


def _get_cutoff_from_index(idx, return_index=False, reverse_order=False):
    """Get cutoff = latest time point of pandas index.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : pd.Index, possibly MultiIndex, with last level assumed timelike or integer,
        e.g., as in the pd.DataFrame, pd-multiindex, or pd_multiindex_hier types
    return_index : bool, optional, default=False
        whether a pd.Index object should be returned (True)
            or a pandas compatible index element (False)
        note: return_index=True may set freq attribute of time types to None
            return_index=False will typically preserve freq attribute
    reverse_order : bool, optional, default=False
        if False, returns largest time index. If True, returns smallest time index

    Returns
    -------
    cutoff_index : pandas compatible index element (if return_index=False)
        pd.Index of length 1 (if return_index=True)
    """
    if not isinstance(idx, pd.Index):
        raise TypeError(f"idx must be a pd.Index, but found type {type(idx)}")

    # define "first" or "last" index depending on which is desired
    if reverse_order:
        ix = 0
        agg = min
    else:
        ix = -1
        agg = max

    if isinstance(idx, pd.MultiIndex):
        tdf = pd.DataFrame(index=idx)
        hix = idx.droplevel(-1)
        freq = None
        cutoff = None
        for hi in hix:
            ss = tdf.loc[hi].index
            if hasattr(ss, "freq") and ss.freq is not None:
                freq = ss.freq
            if cutoff is not None:
                cutoff = agg(cutoff, ss[ix])
            else:
                cutoff = ss[ix]
        time_idx = idx.get_level_values(-1).sort_values()
        time_idx = pd.Index([cutoff])
        time_idx.freq = freq
    else:
        time_idx = idx
        if hasattr(idx, "freq") and idx.freq is not None:
            freq = idx.freq
        else:
            freq = None

    if not return_index:
        return time_idx[ix]
    res = time_idx[[ix]]
    if hasattr(time_idx, "freq") and time_idx.freq is not None:
        res.freq = time_idx.freq
    return res


def get_cutoff(
    obj,
    cutoff=0,
    return_index=False,
    reverse_order=False,
):
    """Get the latest time point of time series or collection of time series.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : aeon compatible time series data container or pandas.Index
        if aeon time series, must be of Series, Collection, or Hierarchical abstract
        type. if ``pandas.Index``, it is assumed that last level is time-like or integer
        e.g., as in the pd.DataFrame, pd-multiindex, or pd_multiindex_hier internal
        types.
    cutoff : int, default=0
        Current cutoff, used to offset index if obj is np.ndarray
    return_index : bool, default=False
        Whether a pd.Index object should be returned (True) or a pandas compatible
        index element (False).
        note: return_index=True may set freq attribute of time types to None
            return_index=False will typically preserve freq attribute.
    reverse_order : bool, default=False
        if False, returns largest time index. If True, returns smallest time index.

    Returns
    -------
    cutoff_index : pandas compatible index element (if return_index=False)
        pd.Index of length 1 (if return_index=True).

    Raises
    ------
    ValueError, TypeError, if check_input or convert_input are True.
    """
    # deal with legacy method of wrapping a Broadcaster
    if hasattr(obj, "X"):
        obj = obj.X

    if isinstance(obj, pd.Index):
        return _get_cutoff_from_index(
            idx=obj, return_index=return_index, reverse_order=reverse_order
        )
    if not (is_hierarchical(obj) or is_collection(obj) or is_single_series(obj)):
        raise ValueError(
            "obj must be of Series, Collection, or Hierarchical abstract type"
        )

    if cutoff is None:
        cutoff = 0
    elif isinstance(cutoff, pd.Index):
        if not len(cutoff) == 1:
            raise ValueError(
                "if cutoff is a pd.Index, its length must be 1, but"
                f" found a pd.Index with length {len(cutoff)}"
            )
        if len(obj) == 0 and return_index:
            return cutoff
        cutoff = cutoff[0]

    if len(obj) == 0:
        return cutoff

    # numpy3D (Collection) or np.npdarray (Series)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3:
            cutoff_ind = obj.shape[-1] + cutoff - 1
        if obj.ndim < 3 and obj.ndim > 0:
            cutoff_ind = obj.shape[0] + cutoff - 1
        if reverse_order:
            cutoff_ind = cutoff
        if return_index:
            return pd.RangeIndex(cutoff_ind, cutoff_ind + 1)
        else:
            return cutoff_ind

    # define "first" or "last" index depending on which is desired
    if reverse_order:
        ix = 0
        agg = min
    else:
        ix = -1
        agg = max

    def sub_idx(idx, ix, return_index=True):
        """Like sub-setting pd.index, but preserves freq attribute."""
        if not return_index:
            return idx[ix]
        res = idx[[ix]]
        if hasattr(idx, "freq") and idx.freq is not None:
            if res.freq != idx.freq:
                res.freq = idx.freq
        return res

    if isinstance(obj, pd.Series):
        return sub_idx(obj.index, ix, return_index)

    # nested_univ (Collection) or pd.DataFrame(Series)
    if isinstance(obj, pd.DataFrame) and not isinstance(obj.index, pd.MultiIndex):
        objcols = [x for x in obj.columns if obj.dtypes[x] == "object"]
        # pd.DataFrame
        if len(objcols) == 0:
            return sub_idx(obj.index, ix) if return_index else obj.index[ix]
        # nested_univ
        else:
            idxx = [
                sub_idx(x.index, ix, return_index) for col in objcols for x in obj[col]
            ]
            return agg(idxx)

    # pd-multiindex (Collection) and pd_multiindex_hier (Hierarchical)
    if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.MultiIndex):
        idx = obj.index
        series_idx = [
            obj.loc[x].index.get_level_values(-1) for x in idx.droplevel(-1).unique()
        ]
        cutoffs = [sub_idx(x, ix, return_index) for x in series_idx]
        return agg(cutoffs)

    # df-list (Collection)
    if isinstance(obj, list):
        idxs = [sub_idx(x.index, ix, return_index) for x in obj]
        return agg(idxs)


SUPPORTED_SERIES = [
    "pd.DataFrame",
    "np.ndarray",
]
SUPPORTED_COLLECTIONS = [
    "pd-multiindex",
    "numpy3D",
]


def update_data(X, X_new=None):
    """Update time series container with another one.

    Converts X, X_new to one of the valid internal types, if not one already.

    Parameters
    ----------
    X : None, or aeon data container, in one of the following internal type formats
        pd.DataFrame, pd.Series, np.ndarray, pd-multiindex, numpy3D,
        pd_multiindex_hier. If not of that format, converted.
    X_new : None, or aeon data container, should be same type as X,
        or convert to same format when converting X

    Returns
    -------
    X updated with X_new, with rows/indices in X_new added
        entries in X_new overwrite X if at same index
        numpy based containers will always be interpreted as having new row index
        if one of X, X_new is None, returns the other; if both are None, returns None
    """
    # Temporary measure to deal with legacy method of wrapping a Broadcaster
    if hasattr(X, "X"):
        X = X.X
    if hasattr(X_new, "X"):
        X_new = X_new.X
    # we only need to modify X if X_new is not None
    if X_new is None:
        return X

    # if X is None, but X_new is not, return N_new
    if X is None:
        return X_new

    # we want to ensure that X, X_new are either numpy (1D, 2D, 3D)
    # or in one of the long pandas formats if they are collections
    if is_collection(X):
        X = convert_collection(X, output_type="numpy3D")
    if is_collection(X_new):
        X_new = convert_collection(X_new, output_type="numpy3D")

    # update X with the new rows in X_new
    #  if X is np.ndarray, we assume all rows are new
    if isinstance(X, np.ndarray):
        # if 1D or 2D, axis 0 is "time"
        if X_new.ndim in [1, 2]:
            return np.concatenate([X, X_new], axis=0)
        # if 3D, axis 2 is "time"
        elif X_new.ndim == 3:
            return np.concatenate([X, X_new], axis=2)
    #  if y is pandas, we use combine_first to update
    elif isinstance(X_new, (pd.Series, pd.DataFrame)) and len(X_new) > 0:
        return X_new.combine_first(X)


def _convert(obj, abstract_type, input_type):
    reconvert = False
    if abstract_type == "Series":
        obj = convert_series(obj, SUPPORTED_SERIES)
        if input_type == "pd.Series":
            reconvert = True
    elif abstract_type == "Panel":
        if input_type not in SUPPORTED_COLLECTIONS:
            obj = convert_collection(obj, "pd-multiindex")
            reconvert = True
    return obj, reconvert


def get_window(obj, window_length=None, lag=None):
    """Slice obj to the time index window with given length and lag.

    Returns time series or time series collection with time indices strictly greater
    than cutoff - lag - window_length, and equal or less than cutoff - lag.
    Cutoff if of obj, as determined by get_cutoff.
    This function does not work with pd.Series, hence the conversion to pd.DataFrame.
    It also does not work with unequal length collections of series.

    Parameters
    ----------
    obj : aeon compatible time series data container or None
        if not None, must be of Series, Collection, or Hierarchical internal types.
        all valid internal types are supported via conversion to internally supported
        types to avoid conversions, pass data in one of SUPPORTED_SERIES or
        SUPPORTED_COLLECTION
    window_length : int or timedelta, optional, default=-inf
        must be int if obj is int indexed, timedelta if datetime indexed
        length of the window to slice to. Default = window of infinite size
    lag : int, timedelta, or None optional, default = None (zero of correct type)
        lag of the latest time in the window, with respect to cutoff of obj
        if None, is internally replaced by a zero of type compatible with obj index
        must be int if obj is int indexed or not pandas based
        must be timedelta if obj is pandas based and datetime indexed

    Returns
    -------
    obj sub-set to time indices in the semi-open interval
        (cutoff - window_length - lag, cutoff - lag)
        None if obj was None
    """
    if obj is None or (window_length is None and lag is None):
        return obj
    valid, metadata = validate_input(obj)
    if not valid:
        raise ValueError("obj must be of Series, Collection, or Hierarchical type")
    input_type = metadata["mtype"]
    abstract_type = metadata["scitype"]
    obj, reconvert = _convert(obj, abstract_type, input_type)

    # numpy3D (Collection) or np.npdarray (Series)
    if isinstance(obj, np.ndarray):
        # if 2D or 3D, we need to subset by last, not first dimension
        # if 1D, we need to subset by first dimension
        # to achieve that effect, we swap first and last in case of 2D, 3D
        # and always subset on first dimension
        if obj.ndim > 1:
            obj = obj.swapaxes(1, -1)
        obj_len = len(obj)
        if lag is None:
            lag = 0
        if window_length is None:
            window_length = obj_len
        window_start = max(-window_length - lag, -obj_len)
        window_end = max(-lag, -obj_len)
        # we need to swap first and last dimension back before returning, if done above
        if window_end == 0:
            obj_subset = obj[window_start:]
        else:
            obj_subset = obj[window_start:window_end]
        if obj.ndim > 1:
            obj_subset = obj_subset.swapaxes(1, -1)
        return obj_subset

    # pd.DataFrame(Series), pd-multiindex (Collection) and pd_multiindex_hier (
    # Hierarchical)
    if isinstance(obj, pd.DataFrame):
        cutoff = get_cutoff(obj)

        if not isinstance(obj.index, pd.MultiIndex):
            time_indices = obj.index
        else:
            time_indices = obj.index.get_level_values(-1)

        if lag is None:
            win_end_incl = cutoff
            win_select = time_indices <= win_end_incl
            if window_length is not None:
                win_start_excl = cutoff - window_length
                win_select = win_select & (time_indices > win_start_excl)
        else:
            win_end_incl = cutoff - lag
            win_select = time_indices <= win_end_incl
            if window_length is not None:
                win_start_excl = cutoff - window_length - lag
                win_select = win_select & (time_indices > win_start_excl)

        obj_subset = obj.iloc[win_select]
        if reconvert:
            if abstract_type == "Series" and input_type == "pd.Series":
                obj_subset = convert_series(obj_subset, input_type)
            elif abstract_type == "Panel":
                obj_subset = convert_collection(obj_subset, input_type)

        return obj_subset

    raise ValueError(
        "passed get_window an object that is not of type np.ndarray or pd.DataFrame"
    )


def get_slice(obj, start=None, end=None):
    """Slice obj with start (inclusive) and end (exclusive) indices.

    Returns time series or time series collection with time indices strictly greater
    and equal to start index and less than end index.

    Parameters
    ----------
    obj : aeon compatible time series data container or None
        if not None, must be of Series, Collection, or Hierarchical type
        in one of SUPPORTED_SERIES or SUPPORTED_COLLECTION.
    start : int or timestamp, optional, default = None
        must be int if obj is int indexed, timestamp if datetime indexed
        Inclusive start of slice. Default = None.
        If None, then no slice at the start
    end : int or timestamp, optional, default = None
        must be int if obj is int indexed, timestamp if datetime indexed
        Exclusive end of slice. Default = None
        If None, then no slice at the end

    Returns
    -------
    obj sub-set sliced for `start` (inclusive) and `end` (exclusive) indices
        None if obj was None
    """
    if (start is None and end is None) or obj is None:
        return obj

    valid, metadata = validate_input(obj)
    if not valid:
        raise ValueError("obj must be of Series, Collection or Hierarchical type")
    input_type = metadata["mtype"]
    abstract_type = metadata["scitype"]
    obj, reconvert = _convert(obj, abstract_type, input_type)

    # numpy3D (Collection) or np.npdarray (Series)
    # Assumes the index is integer so will be exclusive by default
    if isinstance(obj, np.ndarray):
        # if 2D or 3D, we need to subset by last, not first dimension
        # if 1D, we need to subset by first dimension
        # to achieve that effect, we swap first and last in case of 2D, 3D
        # and always subset on first dimension
        if obj.ndim > 1:
            obj = obj.swapaxes(1, -1)
        # subsetting
        if start and end:
            obj_subset = obj[start:end]
        elif end:
            obj_subset = obj[:end]
        else:
            obj_subset = obj[start:]
        # we need to swap first and last dimension back before returning, if done above
        if obj.ndim > 1:
            obj_subset = obj_subset.swapaxes(1, -1)
        return obj_subset

    # pd.DataFrame(Series), pd-multiindex (Collection) and pd_multiindex_hier (
    # Hierarchical)
    # Assumes the index is pd.Timestamp or pd.Period and ensures the end is
    # exclusive with slice_select
    if isinstance(obj, pd.DataFrame):
        if not isinstance(obj.index, pd.MultiIndex):
            time_indices = obj.index
        else:
            time_indices = obj.index.get_level_values(-1)

        if start and end:
            slice_select = (time_indices >= start) & (time_indices < end)
        elif end:
            slice_select = time_indices < end
        elif start:
            slice_select = time_indices >= start

        obj_subset = obj.iloc[slice_select]
        if reconvert:
            if abstract_type == "Series" and input_type == "pd.Series":
                obj_subset = convert_series(obj_subset, input_type)
            elif abstract_type == "Panel":
                obj_subset = convert_collection(obj_subset, input_type)
        return obj_subset
