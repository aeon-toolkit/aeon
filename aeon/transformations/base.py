"""Base class for transformers."""

__maintainer__ = ["MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["BaseTransformer"]

from abc import abstractmethod

from aeon.base import BaseAeonEstimator


class BaseTransformer(BaseAeonEstimator):
    """Transformer base class."""

    _tags = {
        "requires_y": False,
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:missing_values": False,
        "removes_missing_values": False,
    }

    @abstractmethod
    def __init__(self):
        self._estimator_type = "transformer"

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
