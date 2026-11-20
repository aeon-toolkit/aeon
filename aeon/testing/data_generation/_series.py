"""Utility functions for generating series testing data."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "make_example_1d_numpy",
    "make_example_2d_numpy_series",
    "make_example_pandas_series",
    "make_example_dataframe_series",
    "make_example_segment_series",
]


import numpy as np
import pandas as pd

from aeon.testing.data_generation import make_anomaly_detection_labels


def make_example_1d_numpy(
    n_timepoints: int = 12,
    random_state: int | None = None,
    return_y: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Randomly generate 1D numpy X.

    Generates data in 1D 'np.ndarray' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    random_state : int or None, default=None
        Seed for random number generation.
    return_y : str or None, default=None
        If not None, returns an array of relevant labels alongside the generated data.
        Valid options are None and "anomaly".

    Returns
    -------
    X : np.ndarray
        Randomly generated 1D data.
    y : np.ndarray, optional
        If `return_y` is not None, returns an array of relevant labels.

    Raises
    ------
    ValueError
        If `return_y` is not None and not one of the valid options.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_1d_numpy
    >>> data = make_example_1d_numpy(
    ...     n_timepoints=8,
    ...     random_state=0,
    ... )
    >>> print(data)
    [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548  0.64589411
     0.43758721 0.891773  ]
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(size=(n_timepoints,))

    if return_y is not None:
        if return_y == "anomaly" or return_y == "anomaly_detection":
            y = make_anomaly_detection_labels(n_timepoints, random_state=random_state)
            anomaly = [i for i in range(len(y)) if y[i] == 1]
            X[anomaly] = -X[anomaly]
        else:
            raise ValueError(
                f"{return_y} value for return_y is not supported, see the docstring "
                f"for valid options."
            )

        return X, y
    return X


def make_example_2d_numpy_series(
    n_timepoints: int = 12,
    n_channels: int = 1,
    random_state: int | None = None,
    axis: int = 1,
    return_y: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Randomly generate 2D numpy X.

    Generates data in 2D 'np.ndarray' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    n_channels : int, default=1
        The number of series channels to generate.
    random_state : int or None, default=None
        Seed for random number generation.
    axis : int, default=1
        The axis to for the series timepoints. If 1, returns the shape
        (n_channels, n_timepoints). If 0, returns the shape (n_timepoints, n_channels).
    return_y : str or None, default=None
        If not None, returns an array of relevant labels alongside the generated data.
        Valid options are None and "anomaly".

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D data.
    y : np.ndarray, optional
        If `return_y` is not None, returns an array of relevant labels.

    Raises
    ------
    ValueError
        If `return_y` is not None and not one of the valid options.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_2d_numpy_series
    >>> data = make_example_2d_numpy_series(
    ...     n_timepoints=6,
    ...     n_channels=2,
    ...     random_state=0,
    ...     axis=0,
    ... )
    >>> print(data)
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]
     [0.96366276 0.38344152]
     [0.79172504 0.52889492]]
    """
    rng = np.random.RandomState(random_state)
    if axis == 1:
        X = rng.uniform(size=(n_channels, n_timepoints))
    elif axis == 0:
        X = rng.uniform(size=(n_timepoints, n_channels))
    else:
        raise ValueError(f"axis: {axis} is not supported, please use 0 or 1.")

    if return_y is not None:
        if return_y == "anomaly" or return_y == "anomaly_detection":
            y = make_anomaly_detection_labels(n_timepoints, random_state=random_state)
            anomaly = [i for i in range(len(y)) if y[i] == 1]
            if axis == 1:
                X[:, anomaly] = -X[:, anomaly]
            else:
                X[anomaly, :] = -X[anomaly, :]
        else:
            raise ValueError(
                f"{return_y} value for return_y is not supported, see the docstring "
                f"for valid options."
            )

        return X, y
    return X


def make_example_pandas_series(
    n_timepoints: int = 12,
    index_type=None,
    random_state: int | None = None,
    return_y: str | None = None,
) -> pd.Series | tuple[pd.Series, np.ndarray]:
    """Randomly generate pandas Series X.

    Generates data in 'pd.Series' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    index_type : str or None, default=None
        pandas Index type to use. One of ["period", "datetime", "range", "int"].
        If None, uses default integer index.
    random_state : int or None, default=None
        Seed for random number generation.
    return_y : str or None, default=None
        If not None, returns an array of relevant labels alongside the generated data.
        Valid options are None and "anomaly".

    Raises
    ------
    ValueError
        If `return_y` is not None and not one of the valid options.

    Returns
    -------
    X : pd.Series
        Randomly generated 1D data.
    y : np.ndarray, optional
        If `return_y` is not None, returns an array of relevant labels.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_pandas_series
    >>> data = make_example_pandas_series(
    ...     n_timepoints=6,
    ...     random_state=0,
    ... )
    >>> print(data)
    0    0.548814
    1    0.715189
    2    0.602763
    3    0.544883
    4    0.423655
    5    0.645894
    dtype: float64
    """
    rng = np.random.RandomState(random_state)
    index = _make_index(n_timepoints, index_type)
    X = pd.Series(rng.uniform(size=(n_timepoints,)), index=index)

    if return_y is not None:
        if return_y == "anomaly" or return_y == "anomaly_detection":
            y = make_anomaly_detection_labels(n_timepoints, random_state=random_state)
            anomaly = [i for i in range(len(y)) if y[i] == 1]
            X[anomaly] = -X[anomaly]
        else:
            raise ValueError(
                f"{return_y} value for return_y is not supported, see the docstring "
                f"for valid options."
            )

        return X, y
    return X


def make_example_dataframe_series(
    n_timepoints: int = 12,
    n_channels: int = 1,
    index_type=None,
    random_state: int | None = None,
    axis: int = 1,
    return_y: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Randomly generate pandas DataFrame X.

    Generates data in 'pd.DataFrame' format.

    Parameters
    ----------
    n_timepoints : int, default=12
        The number of features/series length to generate.
    n_channels : int, default=1
        The number of series channels to generate.
    index_type : str or None, default=None
        pandas Index type to use. One of ["period", "datetime", "range", "int"].
        If None, uses default integer index.
    random_state : int or None, default=None
        Seed for random number generation.
    axis : int, default=1
        The axis to for the series timepoints. If 1, returns the shape
        (n_channels, n_timepoints). If 0, returns the shape (n_timepoints, n_channels).
    return_y : str or None, default=None
        If not None, returns an array of relevant labels alongside the generated data.
        Valid options are None and "anomaly".

    Returns
    -------
    X : pd.DataFrame
        Randomly generated 2D data.
    y : np.ndarray, optional
        If `return_y` is not None, returns an array of relevant labels.

    Raises
    ------
    ValueError
        If `return_y` is not None and not one of the valid options.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_dataframe_series
    >>> data = make_example_dataframe_series(
    ...     n_timepoints=6,
    ...     n_channels=2,
    ...     random_state=0,
    ...     axis=0,
    ... )
    >>> print(data)
              0         1
    0  0.548814  0.715189
    1  0.602763  0.544883
    2  0.423655  0.645894
    3  0.437587  0.891773
    4  0.963663  0.383442
    5  0.791725  0.528895
    """
    rng = np.random.RandomState(random_state)
    index = _make_index(n_timepoints, index_type)
    if axis == 1:
        X = pd.DataFrame(
            rng.uniform(size=(n_channels, n_timepoints)),
            index=np.arange(n_channels),
            columns=index,
        )
    elif axis == 0:
        X = pd.DataFrame(
            rng.uniform(size=(n_timepoints, n_channels)),
            columns=np.arange(n_channels),
            index=index,
        )
    else:
        raise ValueError(f"axis: {axis} is not supported, please use 0 or 1.")

    if return_y is not None:
        if return_y == "anomaly" or return_y == "anomaly_detection":
            y = make_anomaly_detection_labels(n_timepoints, random_state=random_state)
            anomaly = [i for i in range(len(y)) if y[i] == 1]
            if axis == 1:
                X.iloc[:, anomaly] = -X.iloc[:, anomaly]
            else:
                X.iloc[anomaly, :] = -X.iloc[anomaly, :]
        else:
            raise ValueError(
                f"{return_y} value for return_y is not supported, see the docstring "
                f"for valid options."
            )

        return X, y
    return X


def _make_index(n_timepoints, index_type=None):
    """Make indices for unit testing."""
    if index_type == "period":
        return pd.period_range(start="2000-01", periods=n_timepoints, freq="M")
    elif index_type == "datetime":
        return pd.date_range(start="2000-01-01", periods=n_timepoints, freq="D")
    elif index_type == "range":
        return pd.RangeIndex(start=0, stop=n_timepoints)
    elif index_type == "int" or index_type is None:
        return pd.Index(np.arange(0, n_timepoints), dtype=int)
    else:
        raise ValueError(f"index_class: {index_type} is not supported")


def make_example_segment_series(
    n_timepoints: int = 300,
    n_channels: int = 1,
    shift_type: str = "mean",
    shift_magnitude: float | None = None,
    noise_std: float | None = None,
    axis: int = 1,
    return_changepoints: bool = False,
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""Generate time series with segmented mean or variance shifts.

    Creates a synthetic series divided into three segments:
    - Segment 1 (indices 0 to t1): baseline distribution.
    - Segment 2 (indices t1 to t2): shifted distribution.
    - Segment 3 (indices t2 to n_timepoints): baseline distribution.

    For ``shift_type="mean"``:
    - Segments 1 & 3: :math:`\mathcal{N}(0, \sigma^2)` where
      :math:`\sigma = \text{noise\_std}`.
    - Segment 2: :math:`\mathcal{N}(\mu, \sigma^2)` where
      :math:`\mu = \text{shift\_magnitude}`.

    For ``shift_type="variance"``:
    - Segments 1 & 3: :math:`\mathcal{N}(0, \sigma_1^2)` where
      :math:`\sigma_1 = \text{noise\_std}`.
    - Segment 2: :math:`\mathcal{N}(0, \sigma_2^2)` where
      :math:`\sigma_2 = \text{shift\_magnitude}`.

    Parameters
    ----------
    n_timepoints : int, default=300
        Total length of the time series to generate.
    n_channels : int, default=1
        Number of channels to generate. If 1, returns a 1D array of shape
        ``(n_timepoints,)``. If > 1, returns a 2D array of shape
        ``(n_channels, n_timepoints)`` (if axis=1) or ``(n_timepoints, n_channels)``
        (if axis=0).
    shift_type : {"mean", "variance"}, default="mean"
        Type of distribution change in the middle segment.
    shift_magnitude : float or None, default=None
        Magnitude of the shift in the middle segment. If None, defaults to
        10.0 for ``shift_type="mean"`` and 5.0 for ``shift_type="variance"``.
    noise_std : float or None, default=None
        Standard deviation of the baseline noise in segments 1 and 3.
        If None, defaults to 0.5 for ``shift_type="mean"`` and 1.0 for
        ``shift_type="variance"``.
    axis : int, default=1
        The axis for timepoints when ``n_channels > 1``.
        If 1, shape is ``(n_channels, n_timepoints)``.
        If 0, shape is ``(n_timepoints, n_channels)``.
    return_changepoints : bool, default=False
        If True, also returns the true changepoint locations.
    random_state : int, RandomState, Generator, or None, default=None
        Seed or random state for reproducibility.

    Returns
    -------
    X : np.ndarray
        Synthetic segmented time series.
    changepoints : np.ndarray, optional
        Array of changepoint boundary indices ``[t1, t2]``, returned only if
        ``return_changepoints=True``.

    Raises
    ------
    ValueError
        If ``shift_type`` is not one of ``{"mean", "variance"}``, or if ``axis``
        is not 0 or 1.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_segment_series
    >>> X = make_example_segment_series(n_timepoints=300, random_state=42)
    >>> len(X)
    300
    """
    if axis not in (0, 1):
        raise ValueError(f"axis: {axis} is not supported, please use 0 or 1.")
    if shift_type not in ("mean", "variance"):
        raise ValueError(
            f"shift_type '{shift_type}' is not supported. Must be 'mean' or 'variance'."
        )

    if random_state is None or isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    elif isinstance(random_state, np.random.RandomState):
        rng = np.random.default_rng(random_state.randint(0, 2**31 - 1))
    else:
        raise ValueError(f"Invalid random_state: {random_state}")

    seg1_len = n_timepoints // 3
    seg2_len = n_timepoints // 3
    seg3_len = n_timepoints - seg1_len - seg2_len

    if shift_type == "mean":
        shift = 10.0 if shift_magnitude is None else float(shift_magnitude)
        std = 0.5 if noise_std is None else float(noise_std)
        if n_channels == 1:
            seg1 = rng.normal(0.0, std, seg1_len)
            seg2 = rng.normal(shift, std, seg2_len)
            seg3 = rng.normal(0.0, std, seg3_len)
            X = np.concatenate([seg1, seg2, seg3])
        else:
            seg1 = rng.normal(0.0, std, (n_channels, seg1_len))
            seg2 = rng.normal(shift, std, (n_channels, seg2_len))
            seg3 = rng.normal(0.0, std, (n_channels, seg3_len))
            X = np.hstack([seg1, seg2, seg3])
            if axis == 0:
                X = X.T
    else:  # variance
        std1 = 1.0 if noise_std is None else float(noise_std)
        std2 = 5.0 if shift_magnitude is None else float(shift_magnitude)
        if n_channels == 1:
            seg1 = rng.normal(0.0, std1, seg1_len)
            seg2 = rng.normal(0.0, std2, seg2_len)
            seg3 = rng.normal(0.0, std1, seg3_len)
            X = np.concatenate([seg1, seg2, seg3])
        else:
            seg1 = rng.normal(0.0, std1, (n_channels, seg1_len))
            seg2 = rng.normal(0.0, std2, (n_channels, seg2_len))
            seg3 = rng.normal(0.0, std1, (n_channels, seg3_len))
            X = np.hstack([seg1, seg2, seg3])
            if axis == 0:
                X = X.T

    changepoints = np.array([seg1_len, seg1_len + seg2_len], dtype=np.intp)
    if return_changepoints:
        return X, changepoints
    return X
