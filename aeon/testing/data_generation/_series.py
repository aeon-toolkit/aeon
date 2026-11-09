"""Utility functions for generating series testing data."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "make_example_1d_numpy",
    "make_example_2d_numpy_series",
    "make_example_pandas_series",
    "make_example_dataframe_series",
    "make_example_mean_shift_series",
    "make_example_variance_shift_series",
    "make_example_multivariate_series",
    "make_mean_shift_series",
    "make_variance_shift_series",
    "make_multivariate_series",
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


def make_example_mean_shift_series(
    n_timepoints: int = 300,
    mean_shift: float = 10.0,
    noise_std: float = 0.5,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 1D series with three segments exhibiting mean shifts.

    Creates a 1D numpy array with three segments of roughly equal length.
    The first and third segments have mean 0, and the middle segment has mean
    `mean_shift`. All segments have Gaussian noise with standard deviation `noise_std`.

    Parameters
    ----------
    n_timepoints : int, default=300
        Total length of the time series to generate.
    mean_shift : float, default=10.0
        Mean value of the middle segment.
    noise_std : float, default=0.5
        Standard deviation of Gaussian noise added to all segments.
    random_state : int, np.random.Generator, or None, default=None
        Seed or generator for random number generation.

    Returns
    -------
    X : np.ndarray
        1D numpy array of shape (n_timepoints,).

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_mean_shift_series
    >>> X = make_example_mean_shift_series(n_timepoints=300, random_state=42)
    >>> len(X)
    300
    """
    rng = np.random.default_rng(random_state)
    seg1_len = n_timepoints // 3
    seg2_len = n_timepoints // 3
    seg3_len = n_timepoints - seg1_len - seg2_len

    seg1 = rng.normal(0.0, noise_std, seg1_len)
    seg2 = rng.normal(mean_shift, noise_std, seg2_len)
    seg3 = rng.normal(0.0, noise_std, seg3_len)
    return np.concatenate([seg1, seg2, seg3])


def make_example_variance_shift_series(
    n_timepoints: int = 300,
    scale1: float = 1.0,
    scale2: float = 5.0,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 1D series with three segments exhibiting variance shifts.

    Creates a 1D numpy array with three segments of roughly equal length and zero mean.
    The first and third segments have standard deviation `scale1`, and the middle
    segment has standard deviation `scale2`.

    Parameters
    ----------
    n_timepoints : int, default=300
        Total length of the time series to generate.
    scale1 : float, default=1.0
        Standard deviation of the first and third segments.
    scale2 : float, default=5.0
        Standard deviation of the middle segment.
    random_state : int, np.random.Generator, or None, default=None
        Seed or generator for random number generation.

    Returns
    -------
    X : np.ndarray
        1D numpy array of shape (n_timepoints,).

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_variance_shift_series
    >>> X = make_example_variance_shift_series(n_timepoints=300, random_state=7)
    >>> len(X)
    300
    """
    rng = np.random.default_rng(random_state)
    seg1_len = n_timepoints // 3
    seg2_len = n_timepoints // 3
    seg3_len = n_timepoints - seg1_len - seg2_len

    seg1 = rng.normal(0.0, scale1, seg1_len)
    seg2 = rng.normal(0.0, scale2, seg2_len)
    seg3 = rng.normal(0.0, scale1, seg3_len)
    return np.concatenate([seg1, seg2, seg3])


def make_example_multivariate_series(
    n_timepoints: int = 200,
    n_channels: int = 3,
    mean_shift: float = 8.0,
    noise_std: float = 0.5,
    axis: int = 1,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 2D multivariate series with two segments exhibiting a mean shift.

    Creates a 2D numpy array split into two halves along timepoints.
    The first half has zero mean and the second half has mean `mean_shift` across
    all channels, with Gaussian noise of standard deviation `noise_std`.

    Parameters
    ----------
    n_timepoints : int, default=200
        Number of timepoints to generate.
    n_channels : int, default=3
        Number of channels to generate.
    mean_shift : float, default=8.0
        Mean value of the second segment.
    noise_std : float, default=0.5
        Standard deviation of Gaussian noise.
    axis : int, default=1
        The axis for timepoints. If 1, returns shape (n_channels, n_timepoints).
        If 0, returns shape (n_timepoints, n_channels).
    random_state : int, np.random.Generator, or None, default=None
        Seed or generator for random number generation.

    Returns
    -------
    X : np.ndarray
        2D numpy array of shape (n_channels, n_timepoints) if axis=1,
        or (n_timepoints, n_channels) if axis=0.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_multivariate_series
    >>> X = make_example_multivariate_series(
    ...     n_timepoints=200, n_channels=3, random_state=42
    ... )
    >>> X.shape
    (3, 200)
    """
    if axis not in (0, 1):
        raise ValueError(f"axis: {axis} is not supported, please use 0 or 1.")

    rng = np.random.default_rng(random_state)
    half = n_timepoints // 2
    seg1 = rng.normal(0.0, noise_std, (n_channels, half))
    seg2 = rng.normal(mean_shift, noise_std, (n_channels, n_timepoints - half))
    X = np.hstack([seg1, seg2])

    if axis == 0:
        return X.T
    return X


make_mean_shift_series = make_example_mean_shift_series
make_variance_shift_series = make_example_variance_shift_series
make_multivariate_series = make_example_multivariate_series
