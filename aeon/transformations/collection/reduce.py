"""Tabularizer transform, for pipelining."""

__author__ = ["mloning", "fkiraly", "kcc-lion"]
__all__ = ["Tabularizer", "TimeBinner"]

import warnings

import numpy as np
import pandas as pd

from aeon.datatypes import convert_to
from aeon.transformations.collection import BaseCollectionTransformer


class Tabularizer(BaseCollectionTransformer):
    """
    A transformer that turns time series/panel data into tabular data.

    This estimator converts nested pandas dataframe containing
    time-series/panel data with numpy arrays or pandas Series in
    dataframe cells into a tabular numpy array. This is useful for transforming
    time-series/panel data into a format that is accepted by standard
    validation learning algorithms (as in sklearn).
    """

    _tags = {
        "fit_is_empty": True,
        "output_data_type": "Tabular",
        "X_inner_type": ["numpy3D"],
        "capability:multivariate": True,
    }

    def _transform(self, X, y=None):
        """Transform nested pandas dataframe into tabular dataframe.

        Parameters
        ----------
        X : pandas DataFrame or 3D np.ndarray
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with only primitives in cells.
        """
        Xt = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return Xt


class TimeBinner(BaseCollectionTransformer):
    """
    Turns time series/panel data into tabular data based on intervals.

    This estimator converts nested pandas dataframe containing
    time-series/panel data with numpy arrays or pandas Series in
    dataframe cells into a tabular numpy array.
    The primitives are calculated based on Intervals defined
    by the IntervalIndex and aggregated by aggfunc.

    This is useful for transforming time-series/panel data
    into a format that is accepted by standard validation learning
    algorithms (as in sklearn).

    Parameters
    ----------
    idx : pd.IntervalIndex
        IntervalIndex defining intervals considered by aggfunc.
    aggfunc : callable
        Function used to aggregate the values in intervals.
        Should have signature 1D -> float and defaults
        to mean if None.
    """

    _tags = {
        "fit_is_empty": True,
        "output_data_type": "Tabular",
        "instancewise": True,
        "X_inner_type": ["nested_univ"],
        "y_inner_type": "None",
        "capability:multivariate": True,
    }

    def __init__(self, idx, aggfunc=None):
        assert isinstance(
            idx, pd.IntervalIndex
        ), "idx should be of type pd.IntervalIndex"
        self.aggfunc = aggfunc
        if self.aggfunc is None:
            self._aggfunc = np.mean
            warnings.warn("No aggfunc was passed, defaulting to mean", stacklevel=2)
        else:
            assert callable(aggfunc), (
                "aggfunc should be callable with" "signature 1D -> float"
            )
            if aggfunc.__name__ == "<lambda>":
                warnings.warn(
                    "Save and load will not work with lambda functions", stacklevel=2
                )
            self._aggfunc = self.aggfunc
        self.idx = idx

        super(TimeBinner, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_type
            if X_inner_type is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_type, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        idx = pd.cut(X.iloc[0, 0].index, bins=self.idx, include_lowest=True)
        Xt = X.applymap(lambda x: x.groupby(idx).apply(self._aggfunc))
        Xt = convert_to(Xt, to_type="numpyflat", as_scitype="Panel")
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        import pandas as pd

        idx = pd.interval_range(start=0, end=100, freq=10, closed="left")
        params = {"idx": idx}
        return params
