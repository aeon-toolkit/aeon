"""Base class for estimators that fit single time series.

This time series can be univariate or multivariate. The time series can potentially
contain missing values.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["BaseSeriesEstimator"]

from abc import abstractmethod

import numpy as np
import pandas as pd

from aeon.base._base import BaseAeonEstimator

# allowed input and internal data types for Series
VALID_SERIES_INNER_TYPES = [
    "np.ndarray",
    "pd.DataFrame",
]
VALID_SERIES_INPUT_TYPES = [pd.DataFrame, pd.Series, np.ndarray]


class BaseSeriesEstimator(BaseAeonEstimator):
    """
    Base class for estimators that use single (possibly multivariate) time series.

    Provides functions that are common to estimators which use single series such as
    ``BaseAnomalyDetector``, ``BaseSegmenter``, ``BaseForecaster``,
    and ``BaseSeriesTransformer``. Functionality includes checking and
    conversion of input to ``fit``, ``predict`` and ``predict_proba``, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.

    Input and internal data format (where ``m`` is the number of time points and ``d``
    is the number of channels):
        Univariate series:
            np.ndarray, shape ``(m,)``, ``(m, 1)`` or ``(1, m)`` depending on axis.
            This is converted to a 2D np.ndarray internally.
            pd.DataFrame, shape ``(m, 1)`` or ``(1, m)`` depending on axis.
            pd.Series, shape ``(m,)`` is converted to a pd.DataFrame.
        Multivariate series:
            np.ndarray array, shape ``(m, d)`` or ``(d, m)`` depending on axis.
            pd.DataFrame ``(m, d)`` or ``(d, m)`` depending on axis.

    Parameters
    ----------
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.
        Setting this class variable will convert the input data to the chosen axis.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "X_inner_type": "np.ndarray",  # one of VALID_SERIES_INNER_TYPES
    }

    @abstractmethod
    def __init__(self, axis):
        self.axis = axis
        self.metadata_ = {}  # metadata/properties of data seen in fit

        super().__init__()

    def _preprocess_series(self, X, axis, store_metadata):
        """Preprocess input X prior to call to fit.

        Checks the characteristics of X, store metadata, checks self can handle
        the data then convert X to X_inner_type

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        axis: int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.
        store_metadata: bool
            If True, overwrite metadata with the new metadata from X.

        Returns
        -------
        X: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            Input time series with data structure of type self.get_tag("X_inner_type").
        """
        meta = self._check_X(X, axis)
        if store_metadata:
            self.metadata_ = meta
        return self._convert_X(X, axis)

    def _check_X(self, X, axis):
        """Check input X is valid.

        Check if the input data is a compatible type, and that this estimator is
        able to handle the data characteristics. This is done by matching the
        capabilities of the estimator against the metadata for X for
        univariate/multivariate and no missing values/missing values.

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        axis: int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints,n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels,n_timepoints)``.

        Returns
        -------
        metadata: dict
            Metadata about X, with flags:
            metadata["multivariate"]: whether X has more than one channel or not
            metadata["n_channels"]: number of channels in X
            metadata["missing_values"]: whether X has missing values or not
        """
        if axis > 1 or axis < 0:
            raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

        # Checks: check valid dtype
        if isinstance(X, np.ndarray):
            if not (
                issubclass(X.dtype.type, np.integer)
                or issubclass(X.dtype.type, np.floating)
            ):
                raise ValueError("dtype for np.ndarray must be float or int")
        elif isinstance(X, pd.Series):
            if not pd.api.types.is_numeric_dtype(X):
                raise ValueError("pd.Series dtype must be numeric")
        elif isinstance(X, pd.DataFrame):
            if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                raise ValueError("pd.DataFrame dtype must be numeric")
        else:
            raise ValueError(
                f"Input type of X should be one of {VALID_SERIES_INNER_TYPES}, "
                f"saw {type(X)}"
            )

        metadata = {}

        # check if multivariate
        channel_idx = 0 if axis == 1 else 1
        if X.ndim > 2:
            raise ValueError(
                "X must have at most 2 dimensions for multivariate data, optionally 1 "
                f"for univarate data. Found {X.ndim} dimensions"
            )
        elif X.ndim > 1 and X.shape[channel_idx] > 1:
            metadata["multivariate"] = True
        else:
            metadata["multivariate"] = False

        metadata["n_channels"] = X.shape[channel_idx] if X.ndim > 1 else 1

        # check if has missing values
        if isinstance(X, np.ndarray):
            metadata["missing_values"] = np.isnan(X).any()
        elif isinstance(X, pd.Series):
            metadata["missing_values"] = X.isna().any()
        else:
            metadata["missing_values"] = X.isna().any().any()

        allow_multivariate = self.get_tag("capability:multivariate")
        allow_univariate = self.get_tag("capability:univariate")
        allow_missing = self.get_tag("capability:missing_values")
        if metadata["missing_values"] and not allow_missing:
            raise ValueError(
                f"Missing values not supported by {self.__class__.__name__}"
            )
        if metadata["multivariate"] and not allow_multivariate:
            raise ValueError(
                f"Multivariate data not supported by {self.__class__.__name__}"
            )
        if not metadata["multivariate"] and not allow_univariate:
            raise ValueError(
                f"Univariate data not supported by {self.__class__.__name__}"
            )

        return metadata

    def _convert_X(self, X, axis):
        """Convert input X to internal estimator datatype.

        Converts input X to the internal data type of the estimator using
        self.get_tag("X_inner_type"). 1D numpy arrays are converted to 2D,
        and the data will be transposed if the input axis does not match that of the
        estimator.

        Attempting to convert to a pd.Series for multivariate data or estimators will
        raise an error.

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        axis: int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        X: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            Input time series with data structure of type self.get_tag("X_inner_type").
        """
        if axis > 1 or axis < 0:
            raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

        inner_type = self.get_tag("X_inner_type")
        if not isinstance(inner_type, list):
            inner_type = [inner_type]
        inner_names = [i.split(".")[-1] for i in inner_type]

        input = type(X).__name__
        if input not in inner_names:
            if inner_names[0] == "ndarray":
                X = X.to_numpy()
            elif inner_names[0] == "DataFrame":
                # converting a 1d array will create a 2d array in axis 0 format
                transpose = False
                if X.ndim == 1 and axis == 1:
                    transpose = True
                X = pd.DataFrame(X)
                if transpose:
                    X = X.T
            else:
                raise ValueError(
                    f"Unsupported inner type {inner_names[0]} derived from {inner_type}"
                )

        if X.ndim > 1 and self.axis != axis:
            X = X.T
        elif X.ndim == 1 and isinstance(X, np.ndarray):
            X = X[np.newaxis, :] if self.axis == 1 else X[:, np.newaxis]

        return X
