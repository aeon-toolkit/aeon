"""Abstract base class for time series segmenters."""

__all__ = ["BaseSegmenter"]
__maintainer__ = []

from abc import ABC, abstractmethod
from typing import List, final

import numpy as np

from aeon.base import BaseSeriesEstimator


class BaseSegmenter(BaseSeriesEstimator, ABC):
    """Base class for segmentation algorithms.

    Segmenters take a single time series of length $m$ and returns a segmentation.
    Series can be univariate (single series) or multivariate, with $d$ dimensions.

    Input and internal data format
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

    Conversion and axis resolution for multivariate

        Conversion between numpy and pandas is handled by the base class. Sub classses
        can assume the data is in the correct format (determined by
        ``"X_inner_type"``, one of ``aeon.base._base_series.VALID_INNER_TYPES)`` and
        represented with the expected
        axis.

        Multivariate series are segmented along an axis determined by ``self.axis``.
        Axis plays two roles:

        1) the axis the segmenter expects the data to be in for its internal methods
        ``_fit`` and ``_predict``: 0 means each column is a time series, and the data is
        shaped `(m,d)`, axis equal to 1 means each row is a time series, sometimes
        called wide format, and the whole series is shape `(d,m)`. This should be set
        for a given child class through the BaseSegmenter constructor.

        2) The optional ``axis`` argument passed to the base class ``fit`` and
        ``predict`` methods. If the data ``axis`` is different to the ``axis``
        expected (i.e. value stored in ``self.axis``, then it is transposed in this
        base class if self has multivariate capability.

    Segmentation representation

        Given a time series of 10 points with two change points found in position 4
        and 8.

        The segmentation can be output in two forms:
        a) A list of change points.
            output example [4,8] for a series length 10 means three segments at
            positions (0,1,2,3), (4,5,6,7) and (8,9).
            This dense representation is the default behaviour, as it is the minimal
            representation. Indicated by tag "return_dense" being set to True. It is
            assumed to be sorted, and the first segment is assumed to start at
            position 0. Hence, the first change point must be greater than 0 and the
            last less than the series length. If the last value is
            ``n_timepoints-1`` then the last point forms a single segment. An empty
            list indicates no change points.
        b) A list of integers of length m indicating the segment of each time point:
            output [0,0,0,0,1,1,1,1,2,2] or output [0,0,0,1,1,1,1,0,0,0]
            This sparse representation can be used to indicate shared segments
            indicating segment 1 is somehow the same (perhaps in generative process)
            as segment 3. Indicated by tag ``return_dense`` being set to False.

        Multivariate series are always segmented at the same points. If independent
        segmentation is required, fit a different segmenter to each channel.

    Parameters
    ----------
    n_segments : int, default = 2
        Number of segments to split the time series into. If None, then the number of
        segments needs to be found in fit.
    axis : int, default = 1
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)`.

    """

    _tags = {
        "X_inner_type": "np.ndarray",  # One of VALID_INNER_TYPES
        "fit_is_empty": True,
        "requires_y": False,
        "returns_dense": True,
    }

    def __init__(self, n_segments=2, axis=1):
        self.n_segments = n_segments
        self.axis = axis
        self._is_fitted = False
        super().__init__(axis=axis)

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
        X = self._preprocess_series(X, axis)
        if y is not None:
            y = self._check_y(y)
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
        self.check_is_fitted()
        if axis is None:
            axis = self.axis
        X = self._preprocess_series(X, axis)
        return self._predict(X)

    def fit_predict(self, X, y=None, axis=None):
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

    @classmethod
    def to_classification(cls, change_points: List[int], length: int):
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
    def to_clusters(cls, change_points: List[int], length: int):
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        # default parameters = empty dict
        return {}
