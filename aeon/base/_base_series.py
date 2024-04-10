"""Base class for estimators that fit single time series.

This time series can be univariate or multivariate. The time series can potentially
contain missing values.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["BaseSeriesEstimator"]

import numpy as np
import pandas as pd

from aeon.base._base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps

# allowed input and internal data types for Series
VALID_INNER_TYPES = [
    "np.ndarray",
    "pd.Series",
    "pd.DataFrame",
]
VALID_INPUT_TYPES = [pd.DataFrame, pd.Series, np.ndarray]


class BaseSeriesEstimator(BaseEstimator):
    """Base class for estimators that use single (possibly multivariate) time series.

    Provides functions that are common to BaseSeriesEstimator objects, including
    BaseSeriesTransformer and BaseSegmenter, for the checking and
    conversion of input to fit, predict and transform, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.

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

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)``.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "X_inner_type": "np.ndarray",  # one of VALID_INNER_TYPES
    }

    def __init__(self, axis):
        self.axis = axis
        self.metadata_ = {}  # metadata/properties of data seen in fit

        super().__init__()
        _check_estimator_deps(self)

    def _check_X(self, X, axis):
        """Check input X is valid.

        Check if the input data is a compatible type, and that this estimator is
        able to handle the data characteristics. This is done by matching the
        capabilities of the estimator against the metadata for X for
        univariate/multivariate and no missing values/missing values.

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
           A valid aeon time series data structure. See
           aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis: int
            The time point axis of the input data.

        Returns
        -------
        metadata: dict
            Metadata about X, with flags:
            metadata["multivariate"]: whether X has more than one channel or not
            metadata["n_channels"]: number of channels in X
            metadata["missing_values"]: whether X has missing values or not

        See Also
        --------
        _convert_X: function that converts to inner type.
        """
        if axis > 1 or axis < 0:
            raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

        # Checks: check valid type
        if type(X) not in VALID_INPUT_TYPES:
            raise ValueError(
                f"Input type of X should be one of {VALID_INNER_TYPES}, saw {type(X)}"
            )

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
        elif isinstance(X, pd.DataFrame):
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
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
           A valid aeon time series data structure. See
           aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis: int
            The time point axis of the input data.

        Returns
        -------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
            Input time series with data structure of type self.get_tag("X_inner_type").

        See Also
        --------
        _check_X: function that checks X is valid before conversion.
        """
        if axis > 1 or axis < 0:
            raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

        inner_type = self.get_tag("X_inner_type")
        if not isinstance(inner_type, list):
            inner_type = [inner_type]
        inner_type = [i.split(".")[-1] for i in inner_type]

        input = type(X).__name__
        if input not in inner_type:
            if inner_type[0] == "ndarray":
                X = X.to_numpy()
            elif inner_type[0] == "Series":
                if self.get_tag("capability:multivariate"):
                    raise ValueError(
                        "Cannot convert to pd.Series for multivariate capable "
                        "estimators"
                    )
                if X.ndim > 1:
                    n_channels = X.shape[0] if axis == 1 else X.shape[1]
                    if n_channels > 1:
                        raise ValueError(
                            "Cannot convert to pd.Series for multivariate data. Found "
                            f"{n_channels} channels"
                        )

                X = X.squeeze()
                X = pd.Series(X)
            elif inner_type[0] == "DataFrame":
                # converting a 1d array will create a 2d array in axis 0 format
                transpose = False
                if X.ndim == 1 and axis == 1:
                    transpose = True

                X = pd.DataFrame(X)

                if transpose:
                    X = X.T
            else:
                tag = self.get_tag("X_inner_type")
                raise ValueError(
                    f"Unknown inner type {inner_type[0]} derived from {tag}"
                )

        if X.ndim > 1 and self.axis != axis:
            X = X.T
        elif X.ndim == 1 and isinstance(X, np.ndarray):
            X = X[np.newaxis, :] if self.axis == 1 else X[:, np.newaxis]

        return X

    def _preprocess_series(self, X, axis, store_metadata):
        """Preprocess input X prior to call to fit.

        Checks the characteristics of X, store metadata, checks self can handle
        the data then convert X to X_inner_type

        Parameters
        ----------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
           A valid aeon time series data structure. See
           aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        axis: int or None
            The time point axis of the input data. If None, the default axis is used.
        store_metadata: bool
            If True, overwrite metadata with the new metadata from X.

        Returns
        -------
        X: one of aeon.base._base_series.VALID_INPUT_TYPES
            Input time series with data structure of type self.get_tag("X_inner_type").

        See Also
        --------
        _check_X: function that checks X is valid before conversion.
        _convert_X: function that converts to inner type.
        """
        if axis is None:
            axis = self.axis

        meta = self._check_X(X, axis)
        if store_metadata:
            self.metadata_ = meta

        return self._convert_X(X, axis)
