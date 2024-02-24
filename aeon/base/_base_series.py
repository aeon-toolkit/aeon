"""Base class for estimators that fit single (possibly multivariate) time series."""

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
    axis : int, default = 0
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

    def __init__(self, axis=0):
        self.axis = axis
        self.metadata_ = {}  # metadata/properties of data seen in fit
        super().__init__()
        _check_estimator_deps(self)

    def _check_X(self, X):
        """Check classifier input X is valid.

        Check if the input data is a compatible type, and that this estimator is
        able to handle the data characteristics. This is done by matching the
        capabilities of the estimator against the metadata for X for
        univariate/multivariate, equal length/unequal length and no missing
        values/missing values.

        Parameters
        ----------
        X : data structure
           A valid aeon collection data structure. See
           aeon.registry.COLLECTIONS_DATA_TYPES for details
           on aeon supported data structures.

        Returns
        -------
        dict
            Meta data about X, with flags:
            metadata["missing_values"] : whether X has missing values or not
            metadata["multivariate"] : whether X has more than one channel or not

        See Also
        --------
        _convert_X : function that converts X after it has been checked.
        """
        # Checks: check valid type and axis
        if type(X) not in VALID_INPUT_TYPES:
            raise ValueError(
                f"Error in input type should be one of "
                f" {VALID_INNER_TYPES}, saw {type(X)}"
            )
        if isinstance(X, np.ndarray):
            # Check valid shape
            if X.ndim > 2:
                raise ValueError("Should be 1D or 2D")
            if not (
                issubclass(X.dtype.type, np.integer)
                or issubclass(X.dtype.type, np.floating)
            ):
                raise ValueError("np.ndarray must contain floats or ints")
        elif isinstance(X, pd.Series):
            if not pd.api.types.is_numeric_dtype(X):
                raise ValueError("pd.Series must be numeric")
        else:
            if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                raise ValueError("pd.DataFrame must be numeric")
        # If X is a single series dataframe, we squeeze it into Series in convert_X
        X = X.squeeze()
        metadata = {}
        metadata["multivariate"] = False
        # Need to differentiate because a 1D series stored in a dataframe will have
        # ndim=2. This case is dealt with in convert through squeezing to 1D
        if X.ndim > 1:
            metadata["multivariate"] = True
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
            raise ValueError("Missing values not supported")
        if metadata["multivariate"] and not allow_multivariate:
            raise ValueError("Multivariate data not supported")
        if not metadata["multivariate"] and not allow_univariate:
            raise ValueError("Univariate data not supported")
        return metadata

    def _check_y(self, y: VALID_INPUT_TYPES):
        """Check y specific to segmentation.

        y must be a univariate series
        """
        if type(y) not in VALID_INPUT_TYPES:
            raise ValueError(
                f"Error in input type for y: it should be one of "
                f"{VALID_INPUT_TYPES}, saw {type(y)}"
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

    def _convert_X(self, X, axis):
        inner = self.get_tag("X_inner_type").split(".")[-1]
        input = type(X).__name__
        if inner != input:
            if inner == "ndarray":
                X = X.to_numpy()
            elif inner == "Series":
                if input == "ndarray":
                    X = pd.Series(X)
            elif inner == "DataFrame":
                X = pd.DataFrame(X)
            else:
                tag = self.get_tag("X_inner_type")
                raise ValueError(f"Unknown inner type {inner} derived from {tag}")
        if axis > 1 or axis < 0:
            raise ValueError("Axis should be 0 or 1")
        if not self.get_tag("capability:multivariate"):
            X = X.squeeze()
        elif X.ndim == 1:  # np.ndarray case make 2D
            X = X.reshape(1, -1)
        if X.ndim > 1:
            if self.axis != axis:
                X = X.T
        return X

    def _preprocess_series(self, X, axis=None):
        """Preprocess input X prior to call to fit.

        Checks the characteristics of X, store metadata, checks self can handle
        the data then convert X to X_inner_type

        Parameters
        ----------
        X : one of VALID_INNER_TYPES
        axis: int or None

        Returns
        -------
        Data structure of type self.tags["X_inner_type"]

        See Also
        --------
        _check_X : function that checks X is valid before conversion.
        _convert_X : function that converts to inner type.
        pass
        """
        if axis is None:
            axis = self.axis
        meta = self._check_X(X)
        if len(self.metadata_) == 0:
            self.metadata_ = meta
        return self._convert_X(X, axis)

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
        return {"axis": 0}
