"""Abstract base class for time series segmenters."""

__all__ = ["BaseSegmenter"]
__author__ = ["TonyBagnall"]

from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator

# allowed input and internal data types for Segmenters
ALLOWED_INPUT_TYPES = [
    "ndarray",
    "Series",
    "DataFrame",
]


class BaseSegmenter(BaseEstimator, ABC):
    """Base class for segmentation algorithms.

    Segmenters take a single time series of size $n$ and return a segmentation.
    Series can be multivariate, with $d$ dimensions.

    1. input and internal data format

    Univariate series:
        Numpy array, shape (n,), (n, 1) or (1, n), all converted to (n,)
        pandas Series shape (n,)
        pandas DataFrame single column shape (n,1), (1,n) converted to Series shape (n,)
    Multivariate series:
        Numpy array, shape (n,d) or (d,n)
        pandas DataFrame (n,d) or (d,n)

    2. Conversion and axis resolution for multivariate

    Conversion between numpy and pandas is handled by the base class. Sub classses
    can assume the data is in the correct format (determined by
    "X_inner_type", one of ALLOWED_INPUT_TYPES) and represented with the expected axis.
    Multivariate series are segmented along an axis determined by ``self.axis``. Axis
    plays two roles:
    1) the axis the segmenter expects the data to be in for its internal methods _fit
    and _predict: 0 means each column is a time series channel, 1 means each row is a
    time series channel (sometimes called wide
    format). This should be set for a given child class through the BaseSegmenter
    constructor.
    2) The axis passed to the fit and predict methods. If the data axis is different to
    the axis expected, then it is transposed in this base class.

    3. Segmentation representation

    Given a time series of 10 points with two change points found in position 4 and 8
    (lets index from 1 for clarity)

    The segmentation can be output in three forms:
    a) A list of change points: output example [4,8].
        This dense representation is the default behaviour, as it is the minimal
        representation.
    b) A list of integers of length n indicating the segment of each time point:
        output [0,0,0,1,1,1,1,2,2,2] or output [0,0,0,1,1,1,1,0,0,0]
        This sparse representation can be used to indicate shared segments (indicating
        segment 1 is somehow the same (perhaps in generative process) as segment 3.

    May add others, such as intervals e) intervals [(1,3), (8,10)]

    Segmenters can handle multivariate if the tag "capability:multivariate": is set
    to True.
    Multivariate series are always segmented at the same points. If independent
    segmentation is required, fit a different segmenter to each channel.

    Parameters
    ----------
    n_segments : int, default = 2
        Number of segments to split the time series into. If None, then the number of
        segments needs to be found in fit.
    axis : int, default = 0
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each row is a time series and each column is a channel.
    """

    _tags = {
        "X_inner_type": "ndarray",  # One of ALLOWED_INPUT_TYPES
        "capability:unequal_length": False,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
        "fit_is_empty": True,
    }

    def __init__(self, n_segments=None, axis=1):
        self.n_segments = n_segments
        self.axis = axis
        self._is_fitted = False
        super(BaseSegmenter, self).__init__()

    @final
    def fit(self, X, y=None, axis=None):
        """Fit time series segmenter from training data."""
        # reset estimator at the start of fit
        self.reset()
        if axis is None:  # If none given, assume it is correct.
            axis = self.axis
        self._check_input_series(X)
        X = X.squeeze()  # Remove any single dimensions
        self._check_capabilities(X, axis)
        X = self._convert_X(X, axis)
        if y is not None:
            self._check_y(y)
        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    def predict(self, X, axis=None):
        """Create segmentation."""
        if axis is None:
            axis = self.axis
        self._check_input_series(X)
        self._check_capabilities(X, axis)
        return self._predict(X)

    def fit_predict(self, X, y=None):
        """Fit to data, then predict it."""
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y).predict(X)

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series classifier to training data."""
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predict."""
        ...

    def _check_input_series(self, X):
        """Check input is a data structure only containing floats."""
        # Checks: check valid type and axis
        if not (
            isinstance(X, np.ndarray)
            or isinstance(X, pd.Series)
            or isinstance(X, pd.DataFrame)
        ):
            raise ValueError(
                f" Error in input type should be one onf "
                f" {ALLOWED_INPUT_TYPES}, saw {type(X)}"
            )
        if isinstance(X, np.ndarray):
            # Check valid shape
            if X.ndim > 2:
                raise ValueError(" Should be 1D or 2D")
            if not (
                issubclass(X.dtype.type, np.integer)
                or issubclass(X.dtype.type, np.floating)
            ):
                raise ValueError(" array must contain floats or ints")
        elif isinstance(X, pd.Series):
            if not pd.api.types.is_numeric_dtype(X):
                raise ValueError("pd.Series must be numeric")
        else:
            if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                raise ValueError("pd.DataFrame must be numeric")

    def _check_capabilities(self, X, axis):
        """Check can handle multivariate and missing values."""
        if self.get_tag("capability:multivariate") is False:
            if X.ndim > 1:
                raise ValueError("Multivariate data not supported")

    def _convert_X(self, X, axis):
        # Check axis
        inner = self.get_tag("X_inner_type")
        input = type(X).__name__
        if inner != input:
            if inner == "ndarray":
                X = X.to_numpy()
            elif inner == "Series":
                if input == "ndarray":
                    X = pd.Series(X)
            elif inner == "DataFrame":
                X = pd.DataFrame(X)
        if axis > 1 or axis < 0:
            raise ValueError(" Axis should be 0 or 1")
        if X.ndim > 1:
            if self.axis != axis:
                X = X.T
        return X
