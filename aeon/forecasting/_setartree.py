"""SETAR-Tree: A tree algorithm for global time series forecasting."""

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, solve
from scipy.stats import f
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster

__maintainer__ = ["TinaJin0228"]
__all__ = ["SETARTree"]


class SETARTree(BaseForecaster):
    """
    SETAR-Tree: A tree algorithm for global time series forecasting.

    This implementation is based on the paper "SETAR-Tree: a novel and accurate
    tree algorithm for global time series forecasting" by Godahewa, R., et al. (2023).

    The SETAR-Tree forecaster is a global time series model trained across collections
    of time series, enabling it to learn cross-series patterns.

    Parameters
    ----------
    lag : int, default=10
        The number of past lags to use as features for forecasting.
    max_depth : int, default=10
        The maximum depth of the tree.
    stopping_criteria : {"lin_test", "error_imp", "both"}, default="both"
        The criteria to use for stopping tree growth:
        - "lin_test": Uses a statistical F-test for linearity.
        - "error_imp": Uses a minimum error reduction threshold.
        - "both": Uses both linearity test and error improvement.
    significance : float, default=0.05
        The initial significance level (alpha) for the linearity F-test.
    significance_divider : int, default=2
        The factor by which to divide the significance level at each tree depth.
    error_threshold : float, default=0.03
        The minimum percentage of error reduction required to make a split.
    n_thresholds : int, default=15
        The maximum number of candidate thresholds when trying to find the best
        split point for a single lag.
    scale : bool, default=False
        Whether to scale the time series by their mean before processing.
    seq_significance : bool, default=True
        Whether to decrease significance level with tree depth.
    fixed_lag : bool, default=False
        Whether to use a fixed lag (e.g., Lag{external_lag}) for splitting.
    external_lag : int, default=0
        The specific lag to use if fixed_lag is True.
    """

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": True,
    }

    def __init__(
        self,
        lag=10,
        max_depth=10,
        stopping_criteria="both",
        significance=0.05,
        significance_divider=2,
        error_threshold=0.03,
        n_thresholds=15,
        scale=False,
        seq_significance=True,
        fixed_lag=False,
        external_lag=0,
    ):
        self.lag = lag
        self.max_depth = max_depth
        self.stopping_criteria = stopping_criteria
        self.significance = significance
        self.significance_divider = significance_divider
        self.error_threshold = error_threshold
        self.n_thresholds = n_thresholds
        self.scale = scale
        self.seq_significance = seq_significance
        self.fixed_lag = fixed_lag
        self.external_lag = external_lag
        self.categorical_covariates = []
        self.numerical_covariates = []
        super().__init__(horizon=1, axis=0)

        # Attributes to store fitted model components
        self.tree_ = None
        self.th_lags_ = None
        self.thresholds_ = None
        self.leaf_models_ = None
        self.series_means_ = None
        self.cat_unique_vals_ = {}
        self.final_lags_ = None

    def _embed(self, x, dimension):
        if len(x) < dimension:
            return np.array([]).reshape(0, dimension)
        y = np.array([x[i : i + dimension] for i in range(len(x) - dimension + 1)])
        return y[:, ::-1]

    def _create_tree_input_matrix(self, training_set, test_set=None):
        embedded_rows, final_lags_rows, series_means = [], [], []
        final_cols = [f"Lag{i}" for i in range(1, self.lag + 1)]

        for i in range(len(training_set["series"])):
            time_series = np.asarray(training_set["series"][i], dtype=float)
            mean = np.mean(time_series) if len(time_series) > 0 else 1.0
            series_means.append(mean if mean != 0 else 1.0)

            if self.scale and mean != 0:
                time_series /= mean

            embedded = self._embed(time_series, self.lag + 1)
            if embedded.shape[0] == 0:
                continue

            # No covariates currently
            # covariate_blocks, final_lags_cov_blocks = [], []

            embedded_rows.append(embedded)

            final_lags_ts = time_series[-self.lag :][::-1]
            final_lags_rows.append(final_lags_ts)

        if not embedded_rows:
            return pd.DataFrame(), pd.DataFrame(), []

        full_embedded_matrix = np.vstack(embedded_rows)
        embedded_df = pd.DataFrame(full_embedded_matrix, columns=["y"] + final_cols)
        final_lags_df = pd.DataFrame(np.vstack(final_lags_rows), columns=final_cols)
        return embedded_df, final_lags_df, series_means

    def _find_cut_point(self, X, y, x_ix, k, criterion="RSS"):
        n, p = X.shape
        recheck = 0
        if len(np.unique(x_ix)) <= 1 or np.all(x_ix == x_ix[0]):
            return {"cost": np.inf, "need_recheck": 0}
        q_values = np.linspace(x_ix.min(), x_ix.max(), num=k)
        q = np.concatenate(([-np.inf], q_values, [np.inf]))

        XtX_chunks, Xty_chunks, yty_chunks, n_chunks = [], [], [], []
        XtX_right = np.zeros((p, p))
        Xty_right = np.zeros(p)
        yty_right = 0.0

        for i in range(len(q) - 1):
            ix = (x_ix >= q[i]) & (x_ix < q[i + 1])
            n_s = np.sum(ix)
            n_chunks.append(n_s)
            if n_s == 0:
                XtX_chunks.append(np.zeros((p, p)))
                Xty_chunks.append(np.zeros(p))
                yty_chunks.append(0.0)
            else:
                X_s, y_s = X[ix], y[ix]
                XtX_s = X_s.T @ X_s
                Xty_s = X_s.T @ y_s
                yty_s = np.sum(y_s**2)
                XtX_chunks.append(XtX_s)
                Xty_chunks.append(Xty_s)
                yty_chunks.append(yty_s)
                XtX_right += XtX_s
                Xty_right += Xty_s
                yty_right += yty_s

        XtX_left = np.zeros((p, p))
        Xty_left = np.zeros(p)
        yty_left = 0.0
        n_left, n_right = 0, n
        RSS_left, RSS_right = np.full(k, np.inf), np.full(k, np.inf)
        AICc_left, AICc_right = np.full(k, np.inf), np.full(k, np.inf)

        for i in range(k):
            XtX_left += XtX_chunks[i]
            Xty_left += Xty_chunks[i]
            yty_left += yty_chunks[i]
            n_left += n_chunks[i]
            XtX_right -= XtX_chunks[i]
            Xty_right -= Xty_chunks[i]
            yty_right -= yty_chunks[i]
            n_right -= n_chunks[i]

            if n_left > p + 1 and n_right > p + 1:
                try:
                    b_left = solve(XtX_left, Xty_left, assume_a="pos")
                    rss_l = yty_left - Xty_left @ b_left
                    RSS_left[i] = max(rss_l, 1e-9)
                    if n_left > p + 1:
                        AICc_left[i] = (
                            n_left / 2 * np.log(2 * np.pi * RSS_left[i] / n_left)
                            + n_left / 2
                            + ((p + 1) * n_left) / (n_left - p - 1)
                        )
                    b_right = solve(XtX_right, Xty_right, assume_a="pos")
                    rss_r = yty_right - Xty_right @ b_right
                    RSS_right[i] = max(rss_r, 1e-9)
                    if n_right > p + 1:
                        AICc_right[i] = (
                            n_right / 2 * np.log(2 * np.pi * RSS_right[i] / n_right)
                            + n_right / 2
                            + ((p + 1) * n_right) / (n_right - p - 1)
                        )
                except (LinAlgError, ValueError):
                    recheck += 1
                    continue

        scores = AICc_left + AICc_right if criterion == "AICc" else RSS_left + RSS_right
        if np.all(np.isinf(scores)):
            return {"cost": np.inf, "need_recheck": recheck}
        best_idx = np.nanargmin(scores)
        return {
            "cost": scores[best_idx],
            "cut_point": q_values[best_idx],
            "need_recheck": recheck,
        }

    def _create_split(self, data, conditional_lag_col, threshold):
        left_node = data[data[conditional_lag_col] < threshold]
        right_node = data[data[conditional_lag_col] >= threshold]
        return left_node, right_node

    def _ss(self, p_threshold, train_data, current_lg_col):
        left, right = self._create_split(train_data, current_lg_col, p_threshold)
        if left.empty or right.empty:
            return np.inf
        res_l = self._fit_global_model(left)
        res_r = self._fit_global_model(right)
        residuals_l = left["y"] - res_l["predictions"]
        residuals_r = right["y"] - res_r["predictions"]
        return np.sum(residuals_l**2) + np.sum(residuals_r**2)

    def _check_linearity(self, parent_node, child_nodes, significance):
        parent_model_res = self._fit_global_model(parent_node)
        ss0 = np.sum((parent_node["y"] - parent_model_res["predictions"]) ** 2)
        if ss0 < 1e-9:
            return False
        ss1 = 0
        for child in child_nodes:
            if not child.empty:
                child_model_res = self._fit_global_model(child)
                ss1 += np.sum((child["y"] - child_model_res["predictions"]) ** 2)
        df1 = self.lag + 1
        df2 = parent_node.shape[0] - 2 * self.lag - 2
        if df2 <= 0 or ss1 < 1e-9:
            return False
        f_stat = ((ss0 - ss1) / df1) / (ss1 / df2)
        p_value = 1.0 - f.cdf(f_stat, df1, df2)
        return p_value <= significance

    def _check_error_improvement(self, parent_node, child_nodes):
        parent_model_res = self._fit_global_model(parent_node)
        ss0 = np.sum((parent_node["y"] - parent_model_res["predictions"]) ** 2)
        if ss0 < 1e-9:
            return False
        ss1 = 0
        for child in child_nodes:
            if not child.empty:
                child_model_res = self._fit_global_model(child)
                ss1 += np.sum((child["y"] - child_model_res["predictions"]) ** 2)
        improvement = (ss0 - ss1) / ss0 if ss0 > 0 else 0
        return improvement >= self.error_threshold

    def _get_leaf_index(self, instance, th_lags, thresholds):
        current = 0
        for _level, (lags, threshs) in enumerate(zip(th_lags, thresholds)):
            if lags[current] in (0, None):
                break
            go_right = instance[lags[current]] >= threshs[current]
            splits_before = sum(1 for x in lags[:current] if x not in (0, None))
            pass_before = current - splits_before
            base = splits_before * 2 + pass_before
            current = base + (1 if go_right else 0)
        return current

    def _fit_global_model(self, fitting_data, test_data=None):
        if "y" not in fitting_data.columns:
            raise ValueError("'y' column not found in fitting_data")
        if fitting_data.empty:
            return {"predictions": [], "model": None}
        y = fitting_data["y"]
        X = fitting_data.drop("y", axis=1)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        if test_data is not None:
            X_test = test_data[X.columns]
            predictions = model.predict(X_test)
        else:
            predictions = model.predict(X)
        return {"predictions": predictions, "model": model}

    def _process_input_data(self, y, exog=None):
        training_set = {"series": [y[:, i] for i in range(y.shape[1])]}
        if exog is not None:
            training_set["series"] += [exog[:, i] for i in range(exog.shape[1])]
        return training_set

    def _fit(self, y, exog=None):
        training_set = self._process_input_data(y, exog)
        embedded_series, self.final_lags_, self.series_means_ = (
            self._create_tree_input_matrix(training_set)
        )
        if embedded_series.empty:
            raise ValueError("Failed to create embedded matrix: insufficient data.")

        self.tree_ = []
        self.th_lags_ = []
        self.thresholds_ = []
        node_data = [embedded_series]
        split_info = [1]
        significance = self.significance
        start_con = {"nTh": self.n_thresholds}

        for _d in range(self.max_depth):
            level_th_lags = []
            level_thresholds = []
            level_nodes = []
            level_significant_node_count = 0
            next_level_split_info = []

            for n, current_node_df in enumerate(node_data):
                if (
                    current_node_df.shape[0] <= (2 * (embedded_series.shape[1] - 1) + 2)
                    or split_info[n] == 0
                ):
                    level_th_lags.append(0)
                    level_thresholds.append(0.0)
                    level_nodes.append(current_node_df)
                    next_level_split_info.append(0)
                    continue

                best_cost, best_th, best_th_lag = float("inf"), None, None
                X_features_df = current_node_df.drop("y", axis=1)
                lags_to_check = (
                    [f"Lag{self.external_lag}"]
                    if self.fixed_lag
                    else X_features_df.columns
                )
                X_node = X_features_df.to_numpy()
                y_node = current_node_df["y"].to_numpy()

                for lg_col_name in lags_to_check:
                    lg_col_idx = X_features_df.columns.get_loc(lg_col_name)
                    ss_output = self._find_cut_point(
                        X_node, y_node, X_node[:, lg_col_idx], start_con["nTh"]
                    )
                    if ss_output is None:
                        ss_output = {
                            "cost": float("inf"),
                            "cut_point": None,
                            "need_recheck": 0,
                        }
                    cost = ss_output["cost"]

                    if ss_output["need_recheck"] > round(start_con["nTh"] * 0.6):
                        threshs = np.linspace(
                            current_node_df[lg_col_name].min(),
                            current_node_df[lg_col_name].max(),
                            start_con["nTh"],
                        )
                        for t in threshs:
                            cost_ss = self._ss(t, current_node_df, lg_col_name)
                            if cost_ss < cost:
                                cost = cost_ss
                                ss_output["cut_point"] = t
                    if cost < best_cost:
                        best_cost = cost
                        best_th = ss_output.get("cut_point")
                        best_th_lag = lg_col_name

                is_significant = False
                if best_cost != float("inf") and best_th is not None:
                    split_nodes = self._create_split(
                        current_node_df, best_th_lag, best_th
                    )
                    if not split_nodes[0].empty and not split_nodes[1].empty:
                        if self.stopping_criteria == "lin_test":
                            is_significant = self._check_linearity(
                                current_node_df, split_nodes, significance
                            )
                        elif self.stopping_criteria == "error_imp":
                            is_significant = self._check_error_improvement(
                                current_node_df, split_nodes
                            )
                        elif self.stopping_criteria == "both":
                            is_significant = self._check_linearity(
                                current_node_df, split_nodes, significance
                            ) and self._check_error_improvement(
                                current_node_df, split_nodes
                            )

                if is_significant:
                    level_significant_node_count += 1
                    level_th_lags.append(best_th_lag)
                    level_thresholds.append(best_th)
                    level_nodes.extend(split_nodes)
                    next_level_split_info.extend([1, 1])
                else:
                    level_th_lags.append(0)
                    level_thresholds.append(0.0)
                    level_nodes.append(current_node_df)
                    next_level_split_info.append(0)

            if level_significant_node_count > 0:
                self.tree_.append(level_nodes)
                self.th_lags_.append(level_th_lags)
                self.thresholds_.append(level_thresholds)
                node_data = level_nodes
                split_info = next_level_split_info
                if self.seq_significance:
                    significance /= self.significance_divider
            else:
                break

        leaf_nodes = self.tree_[-1] if self.tree_ else [embedded_series]
        self.leaf_models_ = [
            self._fit_global_model(node)["model"] for node in leaf_nodes
        ]

        return self

    def _predict(self, y=None, exog=None):
        if y is not None:
            training_set = self._process_input_data(y, exog)
            _, current_lags_df, _ = self._create_tree_input_matrix(training_set)
        else:
            current_lags_df = self.final_lags_.copy()

        num_series = current_lags_df.shape[0]
        forecasts = np.zeros((1, num_series))

        horizon_predictions = np.zeros(num_series)
        for i in range(num_series):
            instance = current_lags_df.iloc[i]
            leaf_idx = self._get_leaf_index(instance, self.th_lags_, self.thresholds_)
            leaf_model = self.leaf_models_[leaf_idx]
            horizon_predictions[i] = leaf_model.predict(instance.to_frame().T)[0]
        forecasts[0, :] = horizon_predictions

        if self.scale:
            forecasts *= np.array(self.series_means_)

        return forecasts.flatten() if num_series == 1 else forecasts

    def iterative_forecast(self, y, prediction_horizon):
        # if there are n time series in y, then return n predictions
        if prediction_horizon < 1:
            raise ValueError(
                "The `prediction_horizon` must be greater than or equal to 1."
            )
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_time, n_vars = y.shape
        preds = np.zeros((prediction_horizon, n_vars))
        self.fit(y)
        current_y = y.copy()
        for i in range(prediction_horizon):
            pred = self.predict(current_y)
            preds[i, :] = pred
            current_y = np.vstack((current_y, pred.reshape(1, -1)))
        return preds.squeeze() if n_vars == 1 else preds
