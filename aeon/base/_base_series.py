"""
Base class for single time series estimators (univariate/multivariate).

Class Name: BaseSeriesEstimator

Methods
-------
    - _preprocess_series(X, axis, store_metadata)
      Validates and converts input `X` before fitting.
    - _check_X(X, axis)
      Ensures `X` is a valid type and format.
    - _convert_X(X, axis)
      Converts `X` to the required internal format.

Attributes
----------
    - metadata_
      Stores input series metadata.
    - axis
      Defines time axis for input data.
    - _tags
      Specifies estimator capabilities (e.g., univariate, multivariate).

Inherited Methods:
    - get_params()
      Returns hyperparameters.
    - get_fitted_params()
      Returns learned parameters.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["BaseSeriesEstimator"]

from abc import abstractmethod

from aeon.base._base import BaseAeonEstimator
from aeon.utils.preprocessing import preprocess_series


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
        result = preprocess_series(
            X,
            axis=axis,
            tags=self.get_tags(),
            estimator_axis=self.axis,
            return_metadata=store_metadata,
        )
        if store_metadata:
            X, meta = result
            self.metadata_ = meta
        else:
            X = result
        return X
