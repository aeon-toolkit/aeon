"""SETAR-Tree: A tree algorithm for global time series forecasting."""

import numpy as np
from scipy.linalg import LinAlgError, solve
from scipy.stats import f
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster, IterativeForecastingMixin

__maintainer__ = ["TinaJin0228"]
__all__ = ["SETARTree"]


class SETARTree(BaseForecaster, IterativeForecastingMixin):
    """
    SETAR-Tree: A tree algorithm for global time series forecasting.

    The SETAR-Tree forecaster is a global time series model trained across collections
    of time series, enabling it to learn cross-series patterns.
    This implementation is based on the codebase associated with the work
    by Godahewa et al. [1].

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
    feature_subset : list[int] or None, default=None
        Optional, allowed feature (lag) indices used by the SETARForest.

    References
    ----------
    .. [1] Godahewa, Rakshitha, et al. "SETAR-Tree: a novel and accurate
    tree algorithm for global time series forecasting." Machine Learning
    112.7 (2023): 2555-2591.
    """

    _tags = {
        "capability:horizon": False,
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:univariate": True,
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
        feature_subset=None,
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
        self.feature_subset = feature_subset
        # learned state
        self.tree_ = None
        self.th_lags_ = None  # per depth, list of chosen lag indices
        self.thresholds_ = None  # per depth, list of chosen thresholds
        self.leaf_models_ = None
        self.series_means_ = None
        self.final_lags_ = None
        self.feature_indices_ = None  # training-time feature order

        super().__init__(horizon=1, axis=1)

    def _embed(self, x: np.ndarray, dimension: int) -> np.ndarray:
        """Return reversed sliding windows of length `dimension`."""
        n = len(x)
        if n < dimension:
            return np.empty((0, dimension), dtype=float)
        # rows are x[i:i+dimension], then reverse to [y_t, ..., y_{t-dim+1}]
        M = np.lib.stride_tricks.sliding_window_view(x, window_shape=dimension)
        return M[:, ::-1].astype(float, copy=False)

    def _create_tree_input_matrix(self, training_set: dict, test_set=None):
        """
        Build embedded design from a list of series in `training_set["series"]`.

        Returns
        -------
        embedded : np.ndarray, shape (N, 1+lag)
            First column is y, next columns are Lag1..LagL.
        final_lags : np.ndarray, shape (n_series, lag)
            For each series, the last `lag` values reversed (Lag1..LagL).
        series_means : list[float]
            Mean of each input series (for scaling, if enabled).
        """
        embedded_rows = []
        final_lags_rows = []
        series_means = []

        for s in training_set["series"]:
            x = np.asarray(s, dtype=float)
            mean = np.mean(x) if x.size > 0 else 1.0
            m = mean if mean != 0 else 1.0
            series_means.append(m)

            if self.scale and m != 0:
                x = x / m

            E = self._embed(x, self.lag + 1)
            if E.shape[0] == 0:
                continue

            embedded_rows.append(E)
            final_lags_rows.append(x[-self.lag :][::-1])

        if not embedded_rows:
            return (
                np.empty((0, 1 + self.lag), dtype=float),
                np.empty((0, self.lag), dtype=float),
                [],
            )

        embedded = np.vstack(embedded_rows)
        final_lags = np.vstack(final_lags_rows)
        return embedded, final_lags, series_means

    def _find_cut_point(
        self, X: np.ndarray, y: np.ndarray, x_ix: np.ndarray, k: int, criterion="RSS"
    ):
        n, p = X.shape
        recheck = 0
        if np.all(x_ix == x_ix[0]):
            return {"cost": np.inf, "need_recheck": 0}

        q_values = np.linspace(x_ix.min(), x_ix.max(), num=k)
        q = np.concatenate(([-np.inf], q_values, [np.inf]))

        XtX_chunks, Xty_chunks, yty_chunks, n_chunks = [], [], [], []
        XtX_right = np.zeros((p, p))
        Xty_right = np.zeros(p)
        yty_right = 0.0

        for i in range(len(q) - 1):
            mask = (x_ix >= q[i]) & (x_ix < q[i + 1])
            n_s = int(np.sum(mask))
            n_chunks.append(n_s)
            if n_s == 0:
                XtX_chunks.append(np.zeros((p, p)))
                Xty_chunks.append(np.zeros(p))
                yty_chunks.append(0.0)
            else:
                Xs = X[mask]
                ys = y[mask]
                XtX_s = Xs.T @ Xs
                Xty_s = Xs.T @ ys
                yty_s = float(ys @ ys)
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
                    rss_l = yty_left - float(Xty_left @ b_left)
                    RSS_left[i] = max(rss_l, 1e-9)

                    b_right = solve(XtX_right, Xty_right, assume_a="pos")
                    rss_r = yty_right - float(Xty_right @ b_right)
                    RSS_right[i] = max(rss_r, 1e-9)

                    if criterion == "AICc":
                        AICc_left[i] = (
                            n_left / 2 * np.log(2 * np.pi * RSS_left[i] / n_left)
                            + n_left / 2
                            + ((p + 1) * n_left) / (n_left - p - 1)
                        )
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
        best_idx = int(np.nanargmin(scores))
        return {
            "cost": scores[best_idx],
            "cut_point": q_values[best_idx],
            "need_recheck": recheck,
        }

    def _create_split(self, data: np.ndarray, feat_idx: int, threshold: float):
        """Split node matrix on column (1+feat_idx)."""
        col = 1 + feat_idx
        left = data[data[:, col] < threshold]
        right = data[data[:, col] >= threshold]
        return left, right

    def _ss(self, threshold: float, node: np.ndarray, feat_idx: int) -> float:
        left, right = self._create_split(node, feat_idx, threshold)
        if left.shape[0] == 0 or right.shape[0] == 0:
            return np.inf
        res_l = self._fit_global_model(left)
        res_r = self._fit_global_model(right)
        r_l = left[:, 0] - res_l["predictions"]
        r_r = right[:, 0] - res_r["predictions"]
        return float(r_l @ r_l + r_r @ r_r)

    def _check_linearity(
        self,
        parent: np.ndarray,
        children: tuple[np.ndarray, np.ndarray],
        significance: float,
    ) -> bool:
        res_p = self._fit_global_model(parent)
        ss0 = float(((parent[:, 0] - res_p["predictions"]) ** 2).sum())
        if ss0 < 1e-9:
            return False
        ss1 = 0.0
        for ch in children:
            if ch.shape[0] > 0:
                res_c = self._fit_global_model(ch)
                ss1 += float(((ch[:, 0] - res_c["predictions"]) ** 2).sum())
        df1 = self.lag + 1
        df2 = parent.shape[0] - 2 * self.lag - 2
        if df2 <= 0 or ss1 < 1e-9:
            return False
        f_stat = ((ss0 - ss1) / df1) / (ss1 / df2)
        p_val = 1.0 - f.cdf(f_stat, df1, df2)
        return p_val <= significance

    def _check_error_improvement(
        self, parent: np.ndarray, children: tuple[np.ndarray, np.ndarray]
    ) -> bool:
        res_p = self._fit_global_model(parent)
        ss0 = float(((parent[:, 0] - res_p["predictions"]) ** 2).sum())
        if ss0 < 1e-9:
            return False
        ss1 = 0.0
        for ch in children:
            if ch.shape[0] > 0:
                res_c = self._fit_global_model(ch)
                ss1 += float(((ch[:, 0] - res_c["predictions"]) ** 2).sum())
        improvement = (ss0 - ss1) / ss0 if ss0 > 0 else 0.0
        return improvement >= self.error_threshold

    def _get_leaf_index(
        self,
        instance_feats: np.ndarray,
        th_lags: list[list[int]],
        thresholds: list[list[float]],
    ) -> int:
        """Traverse the tree to get leaf index for a feature vector."""
        current = 0
        for lags, thr in zip(th_lags, thresholds):
            if lags[current] in (0, None):
                break
            go_right = instance_feats[lags[current]] >= thr[current]
            splits_before = sum(1 for x in lags[:current] if x not in (0, None))
            pass_before = current - splits_before
            base = splits_before * 2 + pass_before
            current = base + (1 if go_right else 0)
        return current

    def _fit_global_model(self, data: np.ndarray, test: np.ndarray | None = None):
        """Fit linear regression (no intercept) on node matrix `data`."""
        if data.size == 0:
            return {"predictions": np.array([]), "model": None}
        y = data[:, 0]
        X = data[:, 1:]
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        if test is not None:
            preds = model.predict(test[:, 1:])
        else:
            preds = model.predict(X)
        return {"predictions": preds, "model": model}

    def _process_input_data(self, y: np.ndarray, exog: np.ndarray | None = None):
        """Convert y/exog into dict of series arrays."""
        training_set = {"series": [y[i, :] for i in range(y.shape[0])]}
        if exog is not None:
            training_set["series"].extend([exog[i, :] for i in range(exog.shape[0])])
        return training_set

    def _build_from_embedded(
        self, embedded: np.ndarray, feature_indices: list[int] | None
    ):
        """Build tree and train leaf models from a fully-prepared embedded matrix."""
        if embedded.size == 0:
            raise ValueError("Empty embedded matrix provided.")

        self.feature_indices_ = (
            list(range(embedded.shape[1] - 1))
            if feature_indices is None
            else list(feature_indices)
        )

        self.tree_ = []
        self.th_lags_ = []
        self.thresholds_ = []
        node_data = [embedded]
        split_info = [1]
        significance = self.significance
        nTh = self.n_thresholds

        for _depth in range(self.max_depth):
            level_lags, level_ths, next_nodes = [], [], []
            level_sig_count = 0
            next_split_info = []

            for n, node in enumerate(node_data):
                # minimum rows to attempt a split
                if (
                    node.shape[0] <= (2 * (embedded.shape[1] - 1) + 2)
                    or split_info[n] == 0
                ):
                    level_lags.append(0)
                    level_ths.append(0.0)
                    next_nodes.append(node)
                    next_split_info.append(0)
                    continue

                best_cost, best_th, best_lag = np.inf, None, None
                X = node[:, 1:]
                y = node[:, 0]

                # candidate features (local indices within this embedded feature order)
                p = X.shape[1]
                if self.fixed_lag and self.external_lag > 0:
                    candidates = (
                        [self.external_lag - 1]
                        if 0 < self.external_lag <= p
                        else list(range(p))
                    )
                else:
                    if self.feature_subset is not None:
                        candidates = [i for i in range(p)]
                    else:
                        candidates = list(range(p))

                for j in candidates:
                    out = self._find_cut_point(X, y, X[:, j], nTh, criterion="RSS")
                    cost = out["cost"]
                    if out["need_recheck"] > round(nTh * 0.6):
                        # dense grid fallback
                        lo, hi = X[:, j].min(), X[:, j].max()
                        for t in np.linspace(lo, hi, nTh):
                            c = self._ss(t, node, j)
                            if c < cost:
                                cost = c
                                out["cut_point"] = t
                    if cost < best_cost:
                        best_cost = cost
                        best_th = out.get("cut_point")
                        best_lag = j

                is_sig = False
                if best_cost != np.inf and best_th is not None:
                    left, right = self._create_split(node, best_lag, best_th)
                    if left.shape[0] > 0 and right.shape[0] > 0:
                        if self.stopping_criteria == "lin_test":
                            is_sig = self._check_linearity(
                                node, (left, right), significance
                            )
                        elif self.stopping_criteria == "error_imp":
                            is_sig = self._check_error_improvement(node, (left, right))
                        else:
                            is_sig = self._check_linearity(
                                node, (left, right), significance
                            ) and self._check_error_improvement(node, (left, right))

                if is_sig:
                    level_sig_count += 1
                    level_lags.append(best_lag)
                    level_ths.append(best_th)
                    next_nodes.extend([left, right])
                    next_split_info.extend([1, 1])
                else:
                    level_lags.append(0)
                    level_ths.append(0.0)
                    next_nodes.append(node)
                    next_split_info.append(0)

            if level_sig_count > 0:
                self.tree_.append(next_nodes)
                self.th_lags_.append(level_lags)
                self.thresholds_.append(level_ths)
                node_data = next_nodes
                split_info = next_split_info
                if self.seq_significance:
                    significance /= self.significance_divider
            else:
                break

        leaf_nodes = self.tree_[-1] if self.tree_ else [embedded]
        self.leaf_models_ = [
            self._fit_global_model(leaf)["model"] for leaf in leaf_nodes
        ]
        # As setar-forest will bypass BaseForecaster.fit() when building trees
        self.is_fitted = True
        return self

    def fit_from_embedded(
        self, embedded: np.ndarray, feature_indices: list[int] | None = None
    ):
        """Public hook used by SETARForest: train from a pre-bagged embedded matrix."""
        return self._build_from_embedded(embedded, feature_indices)

    def _fit(self, y, exog=None):
        training_set = self._process_input_data(y, exog)
        embedded, self.final_lags_, self.series_means_ = self._create_tree_input_matrix(
            training_set
        )
        # when fitting directly from y/exog, the feature order is the full lag set
        feature_indices = list(range(embedded.shape[1] - 1)) if embedded.size else None

        self._build_from_embedded(embedded, feature_indices)
        self.forecast_ = self._predict(y)
        return self

    def _predict(self, y, exog=None):
        if not self.leaf_models_:
            raise RuntimeError("SETARTree.predict called before the tree is built.")

        x = y[0, :].astype(float)
        m = np.mean(x) if x.size > 0 else 1.0
        if self.scale and m != 0:
            x = x / m
        if x.size < self.lag:
            raise ValueError("Not enough data points to form required lags.")
        instance_full = x[-self.lag :][::-1]  # Lag1..LagL

        if self.feature_indices_ is None:
            feats = instance_full
        else:
            feats = instance_full[self.feature_indices_]

        # leaf_idx = self._get_leaf_index(feats, self.th_lags_, self.thresholds_)
        # Handle case of zero-depth tree (no splits)
        # where th_lags_/thresholds_ are empty lists
        leaf_idx = (
            0
            if not self.th_lags_
            else self._get_leaf_index(feats, self.th_lags_, self.thresholds_)
        )

        model = self.leaf_models_[leaf_idx]
        pred = float(model.predict(feats.reshape(1, -1))[0])
        if self.scale:
            pred *= float(m if m != 0 else 1.0)
        return pred
