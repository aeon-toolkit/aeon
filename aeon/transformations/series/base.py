"""Base class for Series transformers.

class name: BaseSeriesTransformer

Defining methods:
fitting - fit(self, X, y=None)
transform - transform(self, X, y=None)
fit & transform - fit_transform(self, X, y=None)
"""

from abc import abstractmethod
from typing import final

from deprecated.sphinx import deprecated

from aeon.base import BaseSeriesEstimator
from aeon.transformations.base import BaseTransformer


class BaseSeriesTransformer(BaseSeriesEstimator, BaseTransformer):
    """Transformer base class for collections."""

    # default tag values for series transformers
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
    }

    @abstractmethod
    def __init__(self, axis):
        super().__init__(axis=axis)

    @final
    def fit(self, X, y=None, axis=1):
        """Fit transformer to X, optionally using y if supervised.

        Writes to self:
        - is_fitted : flag is set to True.
        - model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Input data
            Time series to fit transform to, of type ``np.ndarray``, ``pd.Series``
            ``pd.DataFrame``.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
        axis : int, default = 1
            Axis of time in the input series.
            If ``axis == 0``, it is assumed each column is a time series and each row is
            a time point. i.e. the shape of the data is ``(n_timepoints,
            n_channels)``.
            ``axis == 1`` indicates the time series are in rows, i.e. the shape of
            the data is ``(n_channels, n_timepoints)`.``axis is None`` indicates
            that the axis of X is the same as ``self.axis``.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self

        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        # input checks and datatype conversion
        X = self._preprocess_series(X, axis=axis, store_metadata=True)
        if y is not None:
            self._check_y(y)

        self._fit(X=X, y=y)

        self.is_fitted = True
        return self

    @final
    def transform(self, X, y=None, axis=1):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
        axis : int, default = 1
            Axis of time in the input series.
            If ``axis == 0``, it is assumed each column is a time series and each row is
            a time point. i.e. the shape of the data is ``(n_timepoints,
            n_channels)``.
            ``axis == 1`` indicates the time series are in rows, i.e. the shape of
            the data is ``(n_channels, n_timepoints)`.``axis is None`` indicates
            that the axis of X is the same as ``self.axis``.

        Returns
        -------
        transformed version of X with the same axis as passed by the user, if axis
        not None.
        """
        fit_empty = self.get_tag("fit_is_empty")
        if not fit_empty:
            self._check_is_fitted()

        X = self._preprocess_series(X, axis=axis, store_metadata=False)
        if y is not None:
            self._check_y(y)

        # #2768
        # if not fit_empty:
        #     self._check_shape(X)

        Xt = self._transform(X, y)
        return self._postprocess_series(Xt, axis=axis)

    @final
    def fit_transform(self, X, y=None, axis=1):
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
        axis : int, default = 1
            Axis of time in the input series.
            If ``axis == 0``, it is assumed each column is a time series and each row is
            a time point. i.e. the shape of the data is ``(n_timepoints,
            n_channels)``.
            ``axis == 1`` indicates the time series are in rows, i.e. the shape of
            the data is ``(n_channels, n_timepoints)`.``axis is None`` indicates
            that the axis of X is the same as ``self.axis``.

        Returns
        -------
        transformed version of X with the same axis as passed by the user, if axis
        not None.
        """
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        # input checks and datatype conversion
        X = self._preprocess_series(X, axis=axis, store_metadata=True)
        if y is not None:
            self._check_y(y)

        Xt = self._fit_transform(X=X, y=y)

        self.is_fitted = True
        return self._postprocess_series(Xt, axis=axis)

    @final
    def inverse_transform(self, X, y=None, axis=1):
        """Inverse transform X and return an inverse transformed version.

        State required:
             Requires state to be "fitted".

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
             Additional data, e.g., labels for transformation
        axis : int, default = 1
            Axis of time in the input series.
            If ``axis == 0``, it is assumed each column is a time series and each row is
            a time point. i.e. the shape of the data is ``(n_timepoints,
            n_channels)``.
            ``axis == 1`` indicates the time series are in rows, i.e. the shape of
            the data is ``(n_channels, n_timepoints)`.``axis is None`` indicates
            that the axis of X is the same as ``self.axis``.

        Returns
        -------
        inverse transformed version of X
            of the same type as X
        """
        if not self.get_tag("capability:inverse_transform"):
            raise NotImplementedError(
                f"{type(self)} does not implement inverse_transform"
            )

        # check whether is fitted
        self._check_is_fitted()
        X = self._preprocess_series(X, axis=axis, store_metadata=False)
        Xt = self._inverse_transform(X=X, y=y)
        return self._postprocess_series(Xt, axis=axis)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        A default implementation for the fit_is_empty tag. Other transformers
        should override this.

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
        """
        pass

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
        ...

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        private _fit_transform containing the core logic, called from fit_transform.

        Non-optimised default implementation; override when a better
        method is possible for a given algorithm.

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
        if not self.get_tag("fit_is_empty"):
            self._fit(X, y)
        return self._transform(X, y)

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

    def _postprocess_series(self, Xt, axis):
        """Postprocess data Xt to revert to original shape.

        Parameters
        ----------
        Xt: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
            Intended for algorithms which have another series as output.
        axis: int
            The axids of time in the series.
            If  ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.
            If None, the default class axis is used.

        Returns
        -------
        Xt: one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            New time series input reshaped to match the original input.
        """
        if axis is None:
            axis = self.axis

        # If a univariate only transformer, return a univariate series
        if not self.get_tag("capability:multivariate"):
            Xt = Xt.squeeze()

        # return with input axis
        if Xt.ndim == 1 or axis == self.axis:
            return Xt
        else:
            return Xt.T

    # TODO: Remove in v1.4.0
    @deprecated(
        version="1.3.0",
        reason="update is deprecated for transformers and will be removed in v1.4.0.",
        category=FutureWarning,
    )
    @final
    def update(self, X, y=None, update_params=True, axis=1):
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
        self._check_is_fitted()
        X = self._preprocess_series(X, axis, False)
        return self._update(X=X, y=y, update_params=update_params)

    def _update(self, X, y=None, update_params=True):
        # standard behaviour: no update takes place, new data is ignored
        return self
