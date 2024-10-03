"""Base class for transformers."""

__maintainer__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["BaseTransformer"]

from abc import ABC, abstractmethod

from aeon.base import BaseEstimator


class BaseTransformer(BaseEstimator, ABC):
    """Transformer base class."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "transform_labels": "None",
        "instancewise": True,
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "remember_data": False,  # whether all data seen is remembered as self._X
    }

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y=None):
        """Fit transformer to X, optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit transform to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)
        y : Series, default=None
            Additional data, e.g., labels for transformation.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        ...

    @abstractmethod
    def transform(self, X, y=None):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit transform to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)
        y : Series, default=None
            Additional data, e.g., labels for transformation.
        """
        ...

    @abstractmethod
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit transform to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)
        y : Series, default=None
            Additional data, e.g., labels for transformation.
        """
        ...
