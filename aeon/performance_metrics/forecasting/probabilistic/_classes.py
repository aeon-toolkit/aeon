from copy import deepcopy
from logging import warning
from warnings import warn

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils import check_array, check_consistent_length

from aeon.datatypes import VectorizedDF, check_is_scitype, convert, convert_to
from aeon.performance_metrics.base import BaseMetric


class BaseForecastingErrorMetric(BaseMetric):
    """Base class for defining forecasting error metrics in aeon.

    Extends aeon's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values.

    `multioutput` and `multilevel` parameters can be used to control averaging
    across variables (`multioutput`) and (non-temporal) hierarchy levels (`multilevel`).

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', errors are mean-averaged across rows.
        If 'raw_values', does not average errors across levels, hierarchy is retained.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
        "lower_is_better": True,
        # "y_inner_type": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        "inner_implements_multilevel": False,
    }

    def __init__(self, multioutput="uniform_average", multilevel="uniform_average"):
        self.multioutput = multioutput
        self.multilevel = multilevel

        if not hasattr(self, "name"):
            self.name = type(self).__name__
        self.__name__ = self.name
        super().__init__()

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                and self.multilevel="uniform_average" or "uniform_average_time"
                value is metric averaged over variables and levels (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                and self.multilevel="uniform_average" or "uniform_average_time"
                i-th entry is metric calculated for i-th variable
            pd.DataFrame if self.multilevel=raw.values
                of shape (n_levels, ) if self.multioutput = "uniform_average" or array
                of shape (n_levels, y_true.columns) if self.multioutput="raw_values"
                metric is applied per level, row averaging (yes/no) as in multioutput
        """
        return self.evaluate(y_true, y_pred, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                and self.multilevel="uniform_average" or "uniform_average_time"
                value is metric averaged over variables and levels (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                and self.multilevel="uniform_average" or "uniform_average_time"
                i-th entry is metric calculated for i-th variable
            pd.DataFrame if self.multilevel=raw.values
                of shape (n_levels, ) if self.multioutput = "uniform_average" or array
                of shape (n_levels, y_true.columns) if self.multioutput="raw_values"
                metric is applied per level, row averaging (yes/no) as in multioutput
        """
        multioutput = self.multioutput
        multilevel = self.multilevel
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
        )

        requires_vectorization = isinstance(y_true_inner, VectorizedDF)
        if not requires_vectorization:
            # pass to inner function
            out_df = self._evaluate(y_true=y_true_inner, y_pred=y_pred_inner, **kwargs)
        else:
            out_df = self._evaluate_vectorized(
                y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
            )
            if multilevel == "uniform_average":
                out_df = out_df.mean(axis=0)
                # if level is averaged, but not variables, return numpy
                if isinstance(multioutput, str) and multioutput == "raw_values":
                    out_df = out_df.values

        if (
            multilevel == "uniform_average"
            and isinstance(multioutput, str)
            and multioutput == "uniform_average"
        ):
            if isinstance(out_df, pd.DataFrame):
                assert len(out_df) == 1
                assert len(out_df.columns) == 1
                out_df = out_df.iloc[0, 0]
            if isinstance(out_df, pd.Series):
                assert len(out_df) == 1
                out_df = out_df.iloc[0]
        if multilevel == "raw_values":
            out_df = pd.DataFrame(out_df)

        return out_df

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        By default this uses evaluate_by_index, taking arithmetic mean over time points.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                value is metric averaged over variables (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                i-th entry is metric calculated for i-th variable
        """
        # multioutput = self.multioutput
        # multilevel = self.multilevel
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, **kwargs)
            return index_df.mean(axis=0)
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _evaluate_vectorized(self, y_true, y_pred, **kwargs):
        """Vectorized version of _evaluate.

        Runs _evaluate for all instances in y_true, y_pred,
        and returns results in a hierarchical pandas.DataFrame.

        Parameters
        ----------
        y_true : pandas.DataFrame with MultiIndex, last level time-like
        y_pred : pandas.DataFrame with MultiIndex, last level time-like
        non-time-like instanceso of y_true, y_pred must be identical
        """
        kwargsi = deepcopy(kwargs)
        n_batches = len(y_true)
        res = []
        for i in range(n_batches):
            if "y_train" in kwargs:
                kwargsi["y_train"] = kwargs["y_train"][i]
            if "y_pred_benchmark" in kwargs:
                kwargsi["y_pred_benchmark"] = kwargs["y_pred_benchmark"][i]
            resi = self._evaluate(y_true=y_true[i], y_pred=y_pred[i], **kwargsi)
            if isinstance(resi, float):
                resi = pd.Series(resi)
            if self.multioutput == "raw_values":
                assert isinstance(resi, np.ndarray)
                df = pd.DataFrame(columns=y_true.X.columns)
                df.loc[0] = resi
                resi = df
            res += [resi]
        out_df = y_true.reconstruct(res)
        if out_df.index.nlevels == y_true.X.index.nlevels:
            out_df.index = out_df.index.droplevel(-1)

        return out_df

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput
        multilevel = self.multilevel
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
        )
        # pass to inner function
        out_df = self._evaluate_by_index(y_true_inner, y_pred_inner, **kwargs)

        return out_df

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        By default this uses _evaluate to find jackknifed pseudosamples.
        This yields estimates for the metric at each of the time points.
        Caution: this is only sensible for differentiable statistics,
        i.e., not for medians, quantiles or median/quantile based statistics.

        Parameters
        ----------
        y_true : time series in aeon compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series type: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel type: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical type: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in aeon compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput
        n = y_true.shape[0]
        if multioutput == "raw_values":
            out_series = pd.DataFrame(index=y_true.index, columns=y_true.columns)
        else:
            out_series = pd.Series(index=y_true.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, **kwargs)
            for i in range(n):
                idx = y_true.index[i]
                pseudovalue = n * x_bar - (n - 1) * self.evaluate(
                    y_true.drop(idx),
                    y_pred.drop(idx),
                )
                out_series.loc[idx] = pseudovalue
            return out_series
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _check_consistent_input(self, y_true, y_pred, multioutput, multilevel):
        y_true_orig = y_true
        y_pred_orig = y_pred

        # unwrap y_true, y_pred, if wrapped in VectorizedDF
        if isinstance(y_true, VectorizedDF):
            y_true = y_true.X
        if isinstance(y_pred, VectorizedDF):
            y_pred = y_pred.X

        # check row and column indices if y_true vs y_pred
        same_rows = y_true.index.equals(y_pred.index)
        same_row_num = len(y_true.index) == len(y_pred.index)
        same_cols = y_true.columns.equals(y_pred.columns)
        same_col_num = len(y_true.columns) == len(y_pred.columns)

        if not same_row_num:
            raise ValueError("y_pred and y_true do not have the same number of rows.")
        if not same_col_num:
            raise ValueError(
                "y_pred and y_true do not have the same number of columns."
            )

        if not same_rows:
            warn(
                "y_pred and y_true do not have the same row index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred."
            )
            y_pred_orig = y_pred_orig.copy()
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig.X.index = y_true.index
            else:
                y_pred_orig.index = y_true.index
        if not same_cols:
            warn(
                "y_pred and y_true do not have the same column index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred."
            )
            y_pred_orig = y_pred_orig.copy()
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig.X.columns = y_true.columns
            else:
                y_pred_orig.columns = y_true.columns
        # check multioutput arg
        # add this back when variance_weighted is supported
        # ("raw_values", "uniform_average", "variance_weighted")
        allowed_multioutput_str = ("raw_values", "uniform_average")

        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    f"Allowed 'multioutput' values are {allowed_multioutput_str}, "
                    f"but found multioutput={multioutput}"
                )
        else:
            multioutput = check_array(multioutput, ensure_2d=False)
            if len(y_pred.columns) != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), len(y_pred.columns))
                )

        # check multilevel arg
        allowed_multilevel_str = (
            "raw_values",
            "uniform_average",
            "uniform_average_time",
        )

        if not isinstance(multilevel, str):
            raise ValueError(f"multilevel must be a str, but found {type(multilevel)}")
        if multilevel not in allowed_multilevel_str:
            raise ValueError(
                f"Allowed 'multilevel' values are {allowed_multilevel_str}, "
                f"but found multilevel={multilevel}"
            )

        return y_true_orig, y_pred_orig, multioutput, multilevel

    def _check_ys(self, y_true, y_pred, multioutput, multilevel, **kwargs):
        types = ["Series", "Panel", "Hierarchical"]
        inner_types = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]

        def _coerce_to_df(y, var_name="y"):
            valid, msg, metadata = check_is_scitype(
                y, scitype=types, return_metadata=True, var_name=var_name
            )
            if not valid:
                raise TypeError(msg)
            y_inner = convert_to(y, to_type=inner_types)

            type = metadata["scitype"]
            ignore_index = multilevel == "uniform_average_time"
            if type in ["Panel", "Hierarchical"] and not ignore_index:
                y_inner = VectorizedDF(y_inner, is_scitype=type)
            return y_inner

        y_true = _coerce_to_df(y_true, var_name="y_true")
        y_pred = _coerce_to_df(y_pred, var_name="y_pred")
        if "y_train" in kwargs.keys():
            kwargs["y_train"] = _coerce_to_df(kwargs["y_train"], var_name="y_train")
        if "y_pred_benchmark" in kwargs.keys():
            kwargs["y_pred_benchmark"] = _coerce_to_df(
                kwargs["y_pred_benchmark"], var_name="y_pred_benchmark"
            )

        y_true, y_pred, multioutput, multilevel = self._check_consistent_input(
            y_true, y_pred, multioutput, multilevel
        )

        return y_true, y_pred, multioutput, multilevel, kwargs


class _BaseProbaForecastingErrorMetric(BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in aeon.

    Extends aeon's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    score_average : bool, optional, default=True
        for interval and quantile losses only
            if True, metric/loss is averaged by upper/lower and/or quantile
            if False, metric/loss is not averaged by upper/lower and/or quantile
    """

    _tags = {
        "y_input_type_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.multioutput = multioutput
        self.score_average = score_average
        super().__init__(multioutput=multioutput)

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method y_input_type_pred
            must be at fh and for variables equal to those in y_true.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        return self.evaluate(y_true, y_pred, multioutput=self.multioutput, **kwargs)

    def evaluate(self, y_true, y_pred, multioutput=None, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method y_input_type_pred
            must be at fh and for variables equal to those in y_true

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warning(
                "Dropping 0 width interval, don't include 0.5 quantile\
            for interval metrics."
            )

        # pass to inner function
        out = self._evaluate(y_true_inner, y_pred_inner, multioutput, **kwargs)

        if self.score_average and multioutput == "uniform_average":
            out = float(out.mean(axis=1))  # average over all
        elif self.score_average and multioutput == "raw_values":
            out = out.groupby(axis=1, level=0).mean()  # average over scores
        elif not self.score_average and multioutput == "uniform_average":
            out = out.groupby(axis=1, level=1).mean()  # average over variables
        elif not self.score_average and multioutput == "raw_values":
            out = out  # don't average

        if isinstance(out, pd.DataFrame):
            out = out.squeeze(axis=0)

        return out

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.DataFrame or of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.DataFrame of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : pd.DataFrame of shape (, n_outputs), calculated loss metric.
        """
        # Default implementation relies on implementation of evaluate_by_index
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, multioutput)
            out_df = pd.DataFrame(index_df.mean(axis=0)).T
            out_df.columns = index_df.columns
            return out_df
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def evaluate_by_index(self, y_true, y_pred, multioutput=None, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method y_input_type_pred
            must be at fh and for variables equal to those in y_true

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : pd.DataFrame of length len(fh), with calculated metric value(s)
            i-th column contains metric value(s) for prediction at i-th fh element
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warning(
                "Dropping 0 width interval, don't include 0.5 quantile\
            for interval metrics."
            )

        # pass to inner function
        out = self._evaluate_by_index(y_true_inner, y_pred_inner, multioutput, **kwargs)

        if self.score_average and multioutput == "uniform_average":
            out = out.mean(axis=1)  # average over all
        if self.score_average and multioutput == "raw_values":
            out = out.groupby(axis=1, level=0).mean()  # average over scores
        if not self.score_average and multioutput == "uniform_average":
            out = out.groupby(axis=1, level=1).mean()  # average over variables
        if not self.score_average and multioutput == "raw_values":
            out = out  # don't average

        return out

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        By default this uses _evaluate to find jackknifed pseudosamples. This
        estimates the error at each of the time points.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        n = y_true.shape[0]
        out_series = pd.Series(index=y_pred.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
            for i in range(n):
                out_series[i] = n * x_bar - (n - 1) * self.evaluate(
                    np.vstack((y_true[:i, :], y_true[i + 1 :, :])),  # noqa
                    np.vstack((y_pred[:i, :], y_pred[i + 1 :, :])),  # noqa
                    multioutput,
                )
            return out_series
        except RecursionError:
            raise RecursionError(
                "Must implement one of _evaluate or _evaluate_by_index"
            )

    def _check_consistent_input(self, y_true, y_pred, multioutput):
        check_consistent_length(y_true, y_pred)

        y_true = check_array(y_true, ensure_2d=False)

        if not isinstance(y_pred, pd.DataFrame):
            raise ValueError("y_pred should be a dataframe.")

        if not np.all([is_numeric_dtype(y_pred[c]) for c in y_pred.columns]):
            raise ValueError("Data should be numeric.")

        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))

        n_outputs = y_true.shape[1]

        allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    "Allowed 'multioutput' string values are {}. "
                    "You provided multioutput={!r}".format(
                        allowed_multioutput_str, multioutput
                    )
                )
        elif multioutput is not None:
            multioutput = check_array(multioutput, ensure_2d=False)
            if n_outputs == 1:
                raise ValueError("Custom weights are useful only in multi-output case.")
            elif n_outputs != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), n_outputs)
                )

        return y_true, y_pred, multioutput

    def _check_ys(self, y_true, y_pred, multioutput):
        if multioutput is None:
            multioutput = self.multioutput
        valid, msg, metadata = check_is_scitype(
            y_pred, scitype="Proba", return_metadata=True, var_name="y_pred"
        )

        if not valid:
            raise TypeError(msg)

        y_pred_mtype = metadata["mtype"]
        inner_y_pred_mtype = self.get_tag("y_input_type_pred")

        y_pred_inner = convert(
            y_pred,
            from_type=y_pred_mtype,
            to_type=inner_y_pred_mtype,
            as_scitype="Proba",
        )

        if inner_y_pred_mtype == "pred_interval":
            if 0.0 in y_pred_inner.columns.get_level_values(1):
                for var in y_pred_inner.columns.get_level_values(0):
                    y_pred_inner[var, 0.0, "upper"] = y_pred_inner[var, 0.0, "lower"]

        y_pred_inner.sort_index(axis=1, level=[0, 1], inplace=True)

        y_true, y_pred, multioutput = self._check_consistent_input(
            y_true, y_pred, multioutput
        )

        return y_true, y_pred_inner, multioutput

    def _get_alpha_from(self, y_pred):
        """Fetch the alphas present in y_pred."""
        alphas = np.unique(list(y_pred.columns.get_level_values(1)))
        if not all((alphas > 0) & (alphas < 1)):
            raise ValueError("Alpha must be between 0 and 1.")

        return alphas

    def _check_alpha(self, alpha):
        """Check alpha input and coerce to np.ndarray."""
        if alpha is None:
            return None

        if isinstance(alpha, float):
            alpha = [alpha]

        if not isinstance(alpha, np.ndarray):
            alpha = np.asarray(alpha)

        if not all((alpha > 0) & (alpha < 1)):
            raise ValueError("Alpha must be between 0 and 1.")

        return alpha

    def _handle_multioutput(self, loss, multioutput):
        """Specificies how multivariate outputs should be handled.

        Parameters
        ----------
        loss : float, np.ndarray the evaluated metric value.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return loss
            elif multioutput == "uniform_average":
                # pass None as weights to np.average: uniform mean
                multioutput = None
            else:
                raise ValueError(
                    "multioutput is expected to be 'raw_values' "
                    "or 'uniform_average' but we got %r"
                    " instead." % multioutput
                )

        if loss.ndim > 1:
            out = np.average(loss, weights=multioutput, axis=1)
        else:
            out = np.average(loss, weights=multioutput)
        return out


class PinballLoss(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.

    alpha (optional) : float, list or np.ndarray, specifies what quantiles to \
        evaluate metric at.
    """

    _tags = {
        "y_input_type_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True, alpha=None):
        self.score_average = score_average
        self.alpha = alpha
        self._alpha = self._check_alpha(alpha)
        self.metric_args = {"alpha": self._alpha}
        super().__init__(multioutput=multioutput, score_average=score_average)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target value`s.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values"
            Determines how multioutput results will be treated.
        """
        alpha = self._alpha
        y_pred_alphas = self._get_alpha_from(y_pred)
        if alpha is None:
            alphas = y_pred_alphas
        else:
            # if alpha was provided, check whether  they are predicted
            #   if not all alpha are observed, raise a ValueError
            if not np.isin(alpha, y_pred_alphas).all():
                msg = "not all quantile values in alpha are available in y_pred"
                raise ValueError(msg)
            else:
                alphas = alpha

        alphas = self._check_alpha(alphas)

        alpha_preds = y_pred.iloc[
            :, [x in alphas for x in y_pred.columns.get_level_values(1)]
        ]

        alpha_preds_np = alpha_preds.to_numpy()
        alpha_mat = np.repeat(
            (alpha_preds.columns.get_level_values(1).to_numpy().reshape(1, -1)),
            repeats=y_true.shape[0],
            axis=0,
        )

        y_true_np = np.repeat(y_true, axis=1, repeats=len(alphas))
        diff = y_true_np - alpha_preds_np
        sign = (diff >= 0).astype(diff.dtype)
        loss = alpha_mat * sign * diff - (1 - alpha_mat) * (1 - sign) * diff

        out_df = pd.DataFrame(loss, columns=alpha_preds.columns)

        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"alpha": [0.1, 0.5, 0.9]}
        return [params1, params2]


class EmpiricalCoverage(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.
    """

    _tags = {
        "y_input_type_pred": "pred_interval",
        "lower_is_better": False,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.score_average = score_average
        self.multioutput = multioutput
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true
        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        truth_array = (y_true_np > lower).astype(int) * (y_true_np < upper).astype(int)

        out_df = pd.DataFrame(
            truth_array, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        return [params1]


class ConstraintViolation(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.
    """

    _tags = {
        "y_input_type_pred": "pred_interval",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.score_average = score_average
        self.multioutput = multioutput
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true

        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        int_distance = ((y_true_np < lower).astype(int) * abs(lower - y_true_np)) + (
            (y_true_np > upper).astype(int) * abs(y_true_np - upper)
        )

        out_df = pd.DataFrame(
            int_distance, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        return [params1]
