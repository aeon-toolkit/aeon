"""Abstract base class for time series segmenters."""

__all__ = ["BaseSegmenter"]
__author__ = ["TonyBagnall"]

from abc import ABC, abstractmethod
from typing import List, final

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator

# allowed input and internal data types for Segmenters
VALID_INNER_TYPES = [
    "ndarray",
    "Series",
    "DataFrame",
]
VALID_INPUT_TYPES = [pd.DataFrame, pd.Series, np.ndarray]


class BaseSegmenter(BaseEstimator, ABC):
    """Base class for segmentation algorithms.

    Segmenters take a single time series of length $m$ and returns a segmentation.
    Series can be univariate (single series) or multivariate, with $d$ dimensions.

    input and internal data format
        Univariate series:
            Numpy array:
            shape `(m,)`, `(m, 1)` or `(1, m)`. if ``self`` has no multivariate
            capability, i.e.``self.get_tag(
            ""capability:multivariate") == False``, all are converted to 1D
            numpy `(m,)`
            if ``self`` has multivariate capability, converted to 2D numpy `(m,1)` or
            `(1, m)` depending on axis
            pandas DataFrame or Series:
            DataFrame single column shape `(m,1)`, `(1,m)` or Series shape `(m,)`
            if ``self`` has no multivariate capability, all converted to Series `(m,)`
            if ``self`` has multivariate capability, all converted to Pandas DataFrame
            shape `(m,1)`, `(1,m)` depending on axis

    Multivariate series:
        Numpy array, shape `(m,d)` or `(d,m)`.
        pandas DataFrame `(m,d)` or `(d,m)`

    2. Conversion and axis resolution for multivariate

    Conversion between numpy and pandas is handled by the base class. Sub classses
    can assume the data is in the correct format (determined by
    ``"X_inner_type"``, one of ``VALID_INNER_TYPES)`` and represented with the expected
    axis.

    Multivariate series are segmented along an axis determined by ``self.axis``. Axis
    plays two roles:

    1) the axis the segmenter expects the data to be in for its internal methods
    ``_fit`` and ``_predict``: 0 means each column is a time series channel `(m,d)`,
    1 means each row is a time series channel, sometimes called wide
    format, shape `(d,m)`. This should be set for a given child class through the
    BaseSegmenter constructor.


    2) The optional ``axis`` argument passed to the base class ``fit`` and ``predict``
    methods. If the data ``axis`` is different to the ``axis`` expected (i.e. value
    stored in ``self.axis``, then it is htransposed in this base class if self has
    multivariate capability.

    Segmentation representation

    Given a time series of 10 points with two change points found in position 4 and 8
    (lets index from 1 for clarity)

    The segmentation can be output in two forms:
    a) A list of change points: output example [4,8].
        This dense representation is the default behaviour, as it is the minimal
        representation. Indicated by tag "return_dense" being set to True.
    b) A list of integers of length n indicating the segment of each time point:
        output [0,0,0,1,1,1,1,2,2,2] or output [0,0,0,1,1,1,1,0,0,0]
        This sparse representation can be used to indicate shared segments (indicating
        segment 1 is somehow the same (perhaps in generative process) as segment 3.
        Indicated by tag "return_dense" being set to False.

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
        "X_inner_type": "ndarray",  # One of VALID_INNER_TYPES
        "capability:unequal_length": False,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
        "fit_is_empty": True,
        "requires_y": False,
        "returns_dense": True,
    }

    def __init__(self, n_segments=None, axis=1):
        self.n_segments = n_segments
        self.axis = axis
        self._is_fitted = False
        super(BaseSegmenter, self).__init__()

    @final
    def fit(self, X, y=None, axis=None):
        """Fit time series segmenter to X.

        If the tag ``fit_is_empty`` is true, this just sets the ``is_fitted``  tag to
        true. Otherwise, it checks ``self`` can handle ``X``, formats ``X`` into
        the structure required by  ``self`` then passes ``X`` (and possibly ``y``) to
        ``_fit``.

        Parameters
        ----------
        X : One of ``VALID_INPUT_TYPES``
            Input time series
        y : One of ``VALID_INPUT_TYPES`` or None, default None
            Training time series, labeled series same length as X for supervised
            segmentation.
        """
        if self.get_class_tag("fit_is_empty"):
            self._is_fitted = True
            return self
        if self.get_class_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")
        # reset estimator at the start of fit
        self.reset()
        if axis is None:  # If none given, assume it is correct.
            axis = self.axis
        self._check_input_series(X)
        self._check_capabilities(X, axis)
        X = self._convert_series(X, axis)
        if y is not None:
            self._check_input_series(y)
        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    @final
    def predict(self, X, axis=None):
        """Create amd return segmentation of X.

        Parameters
        ----------
        X : One of ``VALID_INPUT_TYPES``
            Input time series
        axis : int, default = None
            Representation of X, ``axis == 0`` indicates ``(n_timepoints,n_channels)``,
            ``axis == 1`` indicates ``(n_channels, n_timepoints)`, ``axis is None``
            indicates that the axis of X is the same as ``self.axis``.

        Returns
        -------
        List
            Either a list of indexes of X indicating where each segment begins or a
            list of integers of ``len(X)`` indicating which segment each time point
            belongs to.
        """
        self.check_is_fitted()
        if axis is None:
            axis = self.axis
        self._check_input_series(X)
        self._check_capabilities(X, axis)
        X = self._convert_series(X, axis)
        return self._predict(X)

    def fit_predict(self, X, y=None):
        """Fit segmentation to data and return it."""
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y).predict(X)

    def _fit(self, X, y):
        """Fit time series classifier to training data."""
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Create and return a segmentation of X."""
        ...

    def _check_input_series(self, X):
        """Check input is one of ``VALID_INPUT_TYPES`` only containing floats."""
        # Checks: check valid type and axis
        if type(X) not in VALID_INPUT_TYPES:
            raise ValueError(
                f" Error in input type should be one onf "
                f" {VALID_INNER_TYPES}, saw {type(X)}"
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
        """Check self can handle multivariate series if X is multivariate."""
        if self.get_class_tag("capability:multivariate") is False:
            if X.ndim > 1:
                raise ValueError("Multivariate data not supported")

    def _convert_series(self, X, axis):
        """Convert X into "X_inner_type" data structure."""
        inner = self.get_class_tag("X_inner_type")
        input = "ndarray"
        if isinstance(X, pd.Series):
            input = "Series"
        elif isinstance(X, pd.DataFrame):
            input = "DataFrame"
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
        if self.get_class_tag("capability:multivariate") and X.ndim == 1:
            X = X.reshape(1, -1)
        else:
            X = X.squeeze()
        if X.ndim > 1:
            if self.axis != axis:
                X = X.T
        return X

    def to_classification(self, change_points: List[int]):
        """Convert change point locations to a classification vector.

        Change point detection results can be treated as classification
        with true change point locations marked with 1's at position of
        the change point and remaining non-change point locations being
        0's.

        For example change points [2, 8] for a time series of length 10
        would result in: [0, 0, 1, 0, 0, 0, 0, 0, 1, 0].
        """
        return np.bincount(change_points[1:-1], minlength=change_points[-1])

    def to_clusters(self, change_points: List[int]):
        """Convert change point locations to a clustering vector.

        Change point detection results can be treated as clustering
        with each segment separated by change points assigned a
        distinct dummy label.

        For example change points [2, 8] for a time series of length 10
        would result in: [0, 0, 1, 1, 1, 1, 1, 1, 2, 2].
        """
        labels = np.zeros(change_points[-1], dtype=np.int32)
        for i, (start, stop) in enumerate(zip(change_points[:-1], change_points[1:])):
            labels[start:stop] = i
        return labels
