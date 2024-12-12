"""Abstract base class for time series segmenters."""

__all__ = ["BaseSegmenter"]
__maintainer__ = []

from abc import abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator
from aeon.base._base_series import VALID_SERIES_INPUT_TYPES


class BaseSegmenter(BaseSeriesEstimator):
    """Base class for segmentation algorithms.

    Segmenters take a single time series of length ``n_timepoints`` and returns a
    segmentation. Series can be univariate (single series) or multivariate,
    with ``n_channels`` dimensions. If the segmenter can handle multivariate series,
    if will have the tag ``"capability:multivariate"`` set to True. Multivariate
    series are segmented along a the axis of time determined by ``self.axis``.

    Segmentation representation

        Given a time series of 10 points with two change points found in position 4
        and 8.

        The segmentation can be output in two forms:
        a) A list of change points (tag ``"returns_dense"`` is True).
            output example [4,8] for a series length 10 means three segments at
            positions (0,1,2,3), (4,5,6,7) and (8,9).
            This dense representation is the default behaviour, as it is the minimal
            representation. Indicated by tag "return_dense" being set to True. It is
            assumed to be sorted, and the first segment is assumed to start at
            position 0. Hence, the first change point must be greater than 0 and the
            last less than the series length. If the last value is
            ``n_timepoints-1`` then the last point forms a single segment. An empty
            list indicates no change points.
        b) A list of integers of length m indicating the segment of each time point (
        tag ``"returns_dense"`` is False).
            output [0,0,0,0,1,1,1,1,2,2] or output [0,0,0,1,1,1,1,0,0,0]
            This sparse representation can be used to indicate shared segments
            indicating segment 1 is somehow the same (perhaps in generative process)
            as segment 3. Indicated by tag ``return_dense`` being set to False.

        Multivariate series are always segmented at the same points. If independent
        segmentation is required, fit a different segmenter to each channel.

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)`. Each segmenter must specify the axis it
        assumes in the constructor and pass it to the base class.
    n_segments : int, default = 2
        Number of segments to split the time series into. If None, then the number of
        segments needs to be found in fit.

    """

    _tags = {
        "X_inner_type": "np.ndarray",  # One of VALID_SERIES_INNER_TYPES
        "fit_is_empty": True,
        "requires_y": False,
        "returns_dense": True,
    }

    @abstractmethod
    def __init__(self, axis, n_segments=2):
        self.n_segments = n_segments

        super().__init__(axis=axis)

    @final
    def fit(self, X, y=None, axis=1):
        """Fit time series segmenter to X.

        If the tag ``fit_is_empty`` is true, this just sets the ``is_fitted``  tag to
        true. Otherwise, it checks ``self`` can handle ``X``, formats ``X`` into
        the structure required by  ``self`` then passes ``X`` (and possibly ``y``) to
        ``_fit``.

        Parameters
        ----------
        X : One of ``VALID_SERIES_INPUT_TYPES``
            Input time series to fit a segmenter.
        y : One of ``VALID_SERIES_INPUT_TYPES`` or None, default None
            Training time series, a labeled 1D series same length as X for supervised
            segmentation.
        axis : int, default = None
            Axis along which to segment if passed a multivariate X series (2D input).
            If axis is 0, it is assumed each column is a time series and each row is
            a time point. i.e. the shape of the data is ``(n_timepoints,
            n_channels)``.
            ``axis == 1`` indicates the time series are in rows, i.e. the shape of
            the data is ``(n_channels, n_timepoints)`.``axis is None`` indicates
            that the axis of X is the same as ``self.axis``.

        Returns
        -------
        self
            Fitted estimator
        """
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")
        # reset estimator at the start of fit
        self.reset()
        if axis is None:  # If none given, assume it is correct.
            axis = self.axis
        X = self._preprocess_series(X, axis, True)
        if y is not None:
            y = self._check_y(y)
        self._fit(X=X, y=y)
        self.is_fitted = True
        return self

    @final
    def predict(self, X, axis=1):
        """Create amd return segmentation of X.

        Parameters
        ----------
        X : One of ``VALID_SERIES_INPUT_TYPES``
            Input time series
        axis : int, default = None
            Axis along which to segment if passed a multivariate series (2D input)
            with ``n_channels`` time series. If axis is 0, it is assumed each row is
            a time series and each column is a time point. i.e. the shape of the data
            is ``(n_timepoints,n_channels)``.
            ``axis == 1`` indicates the time series are in rows, i.e. the shape of
            the data is ``(n_channels, n_timepoints)`.``axis is None`` indicates
            that the axis of X is the same as ``self.axis``.

        Returns
        -------
        List
            Either a list of indexes of X indicating where each segment begins or a
            list of integers of ``len(X)`` indicating which segment each time point
            belongs to.
        """
        self._check_is_fitted()
        if axis is None:
            axis = self.axis
        X = self._preprocess_series(X, axis, False)
        return self._predict(X)

    def fit_predict(self, X, y=None, axis=1):
        """Fit segmentation to data and return it."""
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        self.fit(X, y, axis=axis)
        return self.predict(X, axis=axis)

    def _fit(self, X, y):
        """Fit time series classifier to training data."""
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Create and return a segmentation of X."""
        ...

    def _check_y(self, y: VALID_SERIES_INPUT_TYPES):
        """Check y specific to segmentation.

        y must be a univariate series
        """
        if type(y) not in VALID_SERIES_INPUT_TYPES:
            raise ValueError(
                f"Error in input type for y: it should be one of "
                f"{VALID_SERIES_INPUT_TYPES}, saw {type(y)}"
            )
        if isinstance(y, np.ndarray):
            # Check valid shape
            if y.ndim > 1:
                raise ValueError(
                    "Error in input type for y: y input as np.ndarray " "should be 1D"
                )
            if not (
                issubclass(y.dtype.type, np.integer)
                or issubclass(y.dtype.type, np.floating)
            ):
                raise ValueError(
                    "Error in input type for y: y input must contain " "floats or ints"
                )
        elif isinstance(y, pd.Series):
            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError(
                    "Error in input type for y: y input as pd.Series must be numeric"
                )
        else:  # pd.DataFrame
            if y.shape[1] > 2:
                raise ValueError(
                    "Error in input type for y: y input as pd.DataFrame "
                    "should have a single "
                    "column series"
                )

            if not all(pd.api.types.is_numeric_dtype(y[col]) for col in y.columns):
                raise ValueError(
                    "Error in input type for y: y input as pd.DataFrame "
                    "must be numeric"
                )

    @classmethod
    def to_classification(cls, change_points: list[int], length: int):
        """Convert change point locations to a classification vector.

        Change point detection results can be treated as classification
        with true change point locations marked with 1's at position of
        the change point and remaining non-change point locations being
        0's.

        For example change points [2, 8] for a time series of length 10
        would result in: [0, 0, 1, 0, 0, 0, 0, 0, 1, 0].
        """
        labels = np.zeros(length, dtype=int)
        labels[change_points] = 1
        return labels

    @classmethod
    def to_clusters(cls, change_points: list[int], length: int):
        """Convert change point locations to a clustering vector.

        Change point detection results can be treated as clustering
        with each segment separated by change points assigned a
        distinct dummy label.

        For example change points [2, 8] for a time series of length 10
        would result in: [0, 0, 1, 1, 1, 1, 1, 1, 2, 2].
        """
        labels = np.zeros(length, dtype=int)
        for cp in change_points:
            labels[cp:] += 1
        return labels
