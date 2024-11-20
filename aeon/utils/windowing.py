"""Utility functions for creating and reversing sliding windows of time series data."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["sliding_windows", "reverse_windowing"]

from typing import Callable, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sliding_windows(
    X: np.ndarray, window_size: int, stride: int = 1, axis: int = 1
) -> tuple[np.ndarray, int]:
    """Create sliding windows of a time series.

    Extracts sliding windows of a time series with a given window size and stride. The
    windows are extracted along the specified axis of the time series. If the time
    series is multivariate, the channels of each window are stacked along the
    remaining axis.

    ``stride == 1`` creates default sliding windows, where the window is moved one time
    point at a time. If ``stride > 1``, the windows will skip over some time points.
    If ``stride == window_size``, the windows are non-overlapping, also called tumbling
    windows.

    If the time series length minus the window size is not divisible by the stride, the
    last window will not include the last time points of the time series. The number of
    time points that are not covered by the windows is returned as padding length.

    For a time series with 10 time points, a window size of 3, and a stride of 2, the
    operation will create the following four windows:

        0 7 6 4 4 8 0 6 2 0
        0 7 6
            6 4 4
                4 8 0
                    0 6 2

    The last point at index 9 (0) is not covered by any window. Thus, the padding length
    is 1.

    Parameters
    ----------
    X : np.ndarray
        Univariate (1d) or multivariate (2d) time series with the shape
        (n_timepoints, n_channels).
    window_size : int
        Size of the sliding windows.
    stride : int, optional, default=1
        Stride of the sliding windows. If stride is greater than 1, the windows
        will skip over some time points.
    axis : int, optional, default=1
        Axis along which the sliding windows are extracted. The axis must be the
        time axis of the time series. If ``axis == 1``, ``X`` is assumed to have the
        shape (n_channels, n_timepoints). If ``axis == 0``, ``X`` is assumed to have
        the shape (n_timepoints, n_channels).

    Returns
    -------
    windows : np.ndarray
        Sliding windows of the time series. The return shape depends on the ``axis``
        parameter. If ``axis == 0``, the windows have the shape
        ``(n_windows, window_size)`` (univariate) or
        ``(n_windows, window_size * n_channels)`` (multivariate). If ``axis == 1``, the
        windows have the shape ``(window_size, n_windows)`` (univariate) or
        ``(window_size * n_channels, n_windows)`` (multivariate).
    padding_length : int
        Number of time points that are not covered by the windows. The padding length
        is always 0 for stride 1.

    See Also
    --------
    aeon.utils.windowing.reverse_windowing :
        Reverse windowing of a time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.windowing import sliding_windows
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> windows, _ = sliding_windows(X, window_size=4)
    >>> print(windows)
    [[ 1  2  3  4]
     [ 2  3  4  5]
     [ 3  4  5  6]
     [ 4  5  6  7]
     [ 5  6  7  8]
     [ 6  7  8  9]
     [ 7  8  9 10]]
    >>> windows, padding = sliding_windows(X, window_size=4, stride=4)
    >>> print(windows)
    [[1 2 3 4]
     [5 6 7 8]]
    >>> print(padding)
    2
    """
    if X.ndim == 1:
        axis = 0
    windows = sliding_window_view(
        X, window_shape=window_size, axis=axis, writeable=False
    )
    n_timepoints = X.shape[axis]
    # in case we have a multivariate TS
    if axis == 0:
        windows = windows.reshape((n_timepoints - (window_size - 1), -1))
        windows = windows[::stride, :]
    else:
        windows = windows.reshape((-1, n_timepoints - (window_size - 1)))
        windows = windows[:, ::stride]

    padding_length = n_timepoints - (
        windows.shape[axis] * stride + window_size - stride
    )
    return windows, padding_length


def reverse_windowing(
    y: np.ndarray,
    window_size: int,
    reduction: Callable[..., np.ndarray] = np.nanmean,
    stride: int = 1,
    padding_length: Optional[int] = None,
    force_iterative: bool = False,
) -> np.ndarray:
    """Aggregate windowed results for each point of the original time series.

    Reverse windowing is the inverse operation of sliding windows. It aggregates the
    windowed results for each point of the original time series. The aggregation is
    performed with a reduction function, e.g., np.nanmean, np.nanmedian, or np.nanmax.
    An aggregation function ignoring NaN values is required to handle the padding.

    The resulting time series has the same length as the original time series, namely
    n_timepoints = (n_windows - 1) * stride + window_size + padding_length.

    For a time series of length 10, a window size of 3, and a stride of 2, there are
    four windows. Assuming the following result as input to this function
    `y = np.array([1, 4, 8, 2])`, the example shows the aggregation of the windowed
    results using the np.nanmean reduction function:

        mapped = 1 1 2.5 4 6 8 5 2 2 0
         y[0]  = 1 1  1
         y[1]  =      4  4 4
         y[2]  =           8 8 8
         y[3]  =               2 2 2

    Parameters
    ----------
    y : np.ndarray
        Array of windowed results with the shape (n_windows,).
    window_size : int
        Size of the sliding windows.
    reduction : callable, optional, default=np.nanmean
        Reduction function to aggregate the windowed results. The function must accept
        an axis argument to specify the axis along which the reduction is applied. It
        is required to ignore NaN values to handle the padding on the edges.
    stride : int, optional, default=1
        Stride of the sliding windows. If stride is greater than 1, the windows
        have skipped over some time points. If stride is unequal 1, ``padding_length``
        must be provided as well.
    padding_length : int, optional
        Number of time points at the end of the time series that are not covered by any
        windows. Must be provided if ``stride`` is not 1.
    force_iterative : bool, optional, default=False
        Force iterative reverse windowing function to limit memory usage. If False, the
        function will choose the most efficient reverse windowing function based on the
        available memory trading off between memory usage and speed.

    Returns
    -------
    mapped : np.ndarray
        Array of mapped results with the shape (n_timepoints,). The mapped results are
        aggregated windowed results for each point of the original time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.utils.windowing import sliding_windows, reverse_windowing
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> windows, padding_length = sliding_windows(X, window_size=3, stride=2)
    >>> y = np.array([0.5, 0.6, 0.5, 0.8])
    >>> reverse_windowing(y, window_size=3, stride=2, padding_length=padding_length)
    array([0.5 , 0.5 , 0.55, 0.6 , 0.55, 0.5 , 0.65, 0.8 , 0.8 , 0.  ])
    """
    if stride != 1:
        if padding_length is None:
            raise ValueError("padding_length must be provided when stride is not 1")
        return _reverse_windowing_strided(
            y,
            window_size=window_size,
            stride=stride,
            padding_length=padding_length,
            reduction=reduction,
        )

    if force_iterative:
        return _reverse_windowing_iterative(
            y, window_size=window_size, reduction=reduction
        )

    if _has_enough_memory_for_vectorized_entire(window_size, len(y)):
        return _reverse_windowing_vectorized(
            y, window_size=window_size, reduction=reduction
        )
    else:
        # Falling back to iterative reverse windowing function to limit memory usage!
        return _reverse_windowing_iterative(
            y, window_size=window_size, reduction=reduction
        )


def _has_enough_memory_for_vectorized_entire(window_size: int, n: int) -> bool:
    import sys
    from pathlib import Path

    import psutil

    memory_limit = psutil.virtual_memory().available
    # 128 MB (for other objs) + size of scores array
    memory_buffer = 128 * 1024**2 + sys.getsizeof(float()) * n

    # if we are running within a container
    container_limit_file = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if container_limit_file.exists():
        with container_limit_file.open("r") as fh:
            container_limit = int(fh.read())
        if container_limit < 1 * 1024**4:
            memory_limit = container_limit - memory_buffer

    # if we are running within a dask worker
    try:
        import dask.distributed

        dask_memory_limit = dask.distributed.get_worker().memory_limit
        if dask_memory_limit is not None and dask_memory_limit < memory_limit:
            memory_limit = dask_memory_limit - memory_buffer
    except (ImportError, ValueError):
        pass

    estimated_mem_usage = sys.getsizeof(float()) * (n + window_size - 1) * window_size
    return estimated_mem_usage <= memory_limit


def _reverse_windowing_strided(
    scores: np.ndarray,
    window_size: int,
    stride: int,
    padding_length: int,
    reduction: Callable[..., np.ndarray],
) -> np.ndarray:
    # compute begin and end indices of windows
    begins = np.array([i * stride for i in range(scores.shape[0])], dtype=np.int_)
    ends = begins + window_size

    # prepare target array
    unwindowed_length = stride * (scores.shape[0] - 1) + window_size + padding_length
    mapped = np.full(unwindowed_length, fill_value=np.nan)

    # only iterate over window intersections
    indices = np.unique(np.r_[begins, ends])
    for i, j in zip(indices[:-1], indices[1:]):
        window_indices = np.flatnonzero((begins <= i) & (j - 1 < ends))
        mapped[i:j] = reduction(scores[window_indices])

    # replace untouched indices with 0 (especially for the padding at the end)
    np.nan_to_num(mapped, copy=False)
    return mapped


def _reverse_windowing_vectorized(
    y: np.ndarray, window_size: int, reduction: Callable[..., np.ndarray]
) -> np.ndarray:
    # create big matrix of n x window_size and put each window in its correct place
    unwindowed_length = (window_size - 1) + len(y)
    mapped = np.full(shape=(unwindowed_length, window_size), fill_value=np.nan)
    mapped[: len(y), 0] = y

    for w in range(1, window_size):
        mapped[:, w] = np.roll(mapped[:, 0], w)

    return reduction(mapped, axis=1)


def _reverse_windowing_iterative(
    y: np.ndarray, window_size: int, reduction: Callable[..., np.ndarray]
) -> np.ndarray:
    y = np.array(y, dtype=np.float64)
    unwindowed_length = len(y) + window_size - 1
    pad_n = (window_size - 1, window_size - 1)
    y = np.pad(y, pad_n, "constant", constant_values=(np.nan, np.nan))

    for i in range(len(y) - (window_size - 1)):
        points = y[i : i + window_size]
        y[i] = reduction(points[~np.isnan(points)]).item()

    return y[:unwindowed_length]
