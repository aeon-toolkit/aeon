"""Base class for Series transformers.

class name: BaseSeriesTransformer

Defining methods:
fitting - fit(self, X, y=None)
transform - transform(self, X, y=None)
fit & transform - fit_transform(self, X, y=None)
"""

from abc import ABCMeta, abstractmethod
from typing import final

from aeon.base import BaseSeriesEstimator
from aeon.transformations.base import BaseTransformer


class BaseSeriesTransformer(BaseSeriesEstimator, BaseTransformer, metaclass=ABCMeta):
    """Transformer base class for collections."""

    # tag values specific to SeriesTransformers
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "X_inner_type": "ndarray",
        "fit_is_empty": False,
        "requires_y": False,
        "capability:inverse_transform": False,
    }

    def __init__(self, axis=1):
        self.axis = axis
        super().__init__(axis=axis)

    @final
    def fit(self, X, y=None, axis=None):
        """Fit transformer to X, optionally using y if supervised.

        State change:
            Changes state to "fitted".

        Parameters
        ----------
        X : Input data
            Time series to fit transform to, of type ``np.ndarray``, ``pd.Series``
            ``pd.DataFrame``.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
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
        self : a fitted instance of the estimator
        """
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")
        # skip the rest if fit_is_empty is True
        if self.get_tag("fit_is_empty"):
            self._is_fitted = True
            return self
        # reset estimator at the start of fit
        self.reset()
        if axis is None:  # If none given, assume it is correct.
            axis = self.axis
        X = self._preprocess_series(X, axis=axis)
        if y is not None:
            self._check_y(y)
        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    @final
    def transform(self, X, axis=None):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
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
        transformed version of X
        """
        # check whether is fitted
        self.check_is_fitted()

        if axis is None:
            axis = self.axis
        X = self._preprocess_series(X, axis=axis)
        return self._transform(X)

    @final
    def fit_transform(self, X, y=None, axis=None):
        """
        Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        Changes state to "fitted". Model attributes (ending in "_") : dependent on
        estimator.

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
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
        transformed version of X
        """
        # input checks and datatype conversion, to avoid doing in both fit and transform
        self.reset()
        X = self._preprocess_series(X, axis=axis)
        Xt = self._fit_transform(X=X, y=y)
        self._is_fitted = True
        return Xt

    @final
    def inverse_transform(self, X, y=None, axis=None):
        """Inverse transform X and return an inverse transformed version.

        State required:
             Requires state to be "fitted".

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
             Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X
        """
        if self.get_tag("skip-inverse-transform"):
            return X

        if not self.get_tag("capability:inverse_transform"):
            raise NotImplementedError(
                f"{type(self)} does not implement inverse_transform"
            )

        # check whether is fitted
        self.check_is_fitted()
        X = self._preprocess_series(X, axis=axis)
        return self._inverse_transform(X=X, y=y)

    @final
    def update(self, X, y=None, update_params=True, axis=None):
        """Update transformer with X, optionally y.

        Parameters
        ----------
        X : data to update of valid series type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
        update_params : bool, default=True
            whether the model is updated. Yes if true, if false, simply skips call.
            argument exists for compatibility with forecasting module.
        axis : int, default=None
            axis along which to update. If None, uses self.axis.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # check whether is fitted
        self.check_is_fitted()
        X = self._preprocess_series(X, axis=axis)
        return self._update(X=X, y=y, update_params=update_params)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        # default fit is "no fitting happens"
        return self

    @abstractmethod
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        private _fit_transform containing the core logic, called from fit_transform.

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation.

        Returns
        -------
        transformed version of X.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self._fit(X, y)._transform(X, y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform.

        Parameters
        ----------
        X : Input data
            Time series to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def _update(self, X, y=None):
        # standard behaviour: no update takes place, new data is ignored
        return self

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
        return {}
