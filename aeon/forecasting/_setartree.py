"""SETAR-Tree: A tree algorithm for global time series forecasting."""

import numpy as np
from scipy.stats import f
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster

__maintainer__ = ["TinaJin0228"]
__all__ = ["SetartreeForecaster"]


class SetartreeForecaster(BaseForecaster):
    """
    SETAR-Tree: A tree algorithm for global time series forecasting.

    This implementation is based on the paper "SETAR-Tree: a novel and accurate
    tree algorithm for global time series forecasting" by Godahewa, R., et al. (2023).

    Parameters
    ----------
    lag : int, default=10
        The number of past lags to use as features for forecasting.
    horizon : int, default=1
        The number of time steps ahead to forecast.
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
    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        lag: int = 10,
        horizon: int = 1,
        max_depth: int = 10,
        stopping_criteria: str = "both",
        significance: float = 0.05,
        significance_divider: int = 2,
        error_threshold: float = 0.03,
    ):
        super().__init__(horizon=horizon, axis=1)
        self.lag = lag
        self.max_depth = max_depth
        self.stopping_criteria = stopping_criteria
        self.significance = significance
        self.significance_divider = significance_divider
        self.error_threshold = error_threshold

        # Attributes to be fitted
        self.tree_ = {}
        self.leaf_models_ = {}

        # for in-sample prediction when y = None in predict()
        self._last_window = None

    def _create_input_matrix(self, y: np.ndarray):
        """Create an embedded matrix from a time series."""
        n_series, n_timepoints = y.shape

        # We need at least lag + 1 points to create one sample
        if n_timepoints < self.lag + 1:
            return None, None

        X_list, y_list = [], []
        for i in range(n_series):
            series = y[i, :]
            for j in range(len(series) - self.lag):
                X_list.append(series[j : j + self.lag])
                y_list.append(series[j + self.lag])

        # The paper uses a convention of Lags 1, 2...
        # we flip the columns to match the convention
        return np.fliplr(np.array(X_list)), np.array(y_list)

    def _fit_pr_model(self, X, y):
        """Fit a Pooled Regression (linear) model."""
        model = LinearRegression()
        model.fit(X, y)
        return model

    def _calculate_sse(self, model, X, y):
        """Calculate the Sum of Squared Errors (SSE)."""
        if len(y) == 0:
            return 0
        predictions = model.predict(X)
        return np.sum((y - predictions) ** 2)

    def _find_optimal_split(self, X, y):
        """Find the best lag and threshold for a split."""
        # currently brute-force grid search method
        best_split = {"sse": float("inf")}
        n_samples, n_features = X.shape

        for lag_idx in range(n_features):
            thresholds = np.unique(X[:, lag_idx])
            for t in thresholds:
                left_indices = X[:, lag_idx] < t
                right_indices = X[:, lag_idx] >= t

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                X_left, y_left = X[left_indices], y[left_indices]
                X_right, y_right = X[right_indices], y[right_indices]

                model_left = self._fit_pr_model(X_left, y_left)
                model_right = self._fit_pr_model(X_right, y_right)

                sse_left = self._calculate_sse(model_left, X_left, y_left)
                sse_right = self._calculate_sse(model_right, X_right, y_right)
                total_sse = sse_left + sse_right

                if total_sse < best_split["sse"]:
                    best_split = {
                        "sse": total_sse,
                        "lag_idx": lag_idx,
                        "threshold": t,
                        "left_indices": left_indices,
                        "right_indices": right_indices,
                    }
        return best_split

    def _check_linearity(self, parent_sse, child_sse, n_samples, current_alpha):
        """Perform the F-test to check for remaining non-linearity."""
        # Degrees of freedom for parent and child models
        # df_parent = n_samples - self.lag - 1
        df_child = n_samples - 2 * self.lag - 2

        if df_child <= 0:
            return False

        f_stat = ((parent_sse - child_sse) / (self.lag + 1)) / (child_sse / df_child)
        p_value = 1 - f.cdf(f_stat, self.lag + 1, df_child)

        return p_value < current_alpha

    def _check_error_improvement(self, parent_sse, child_sse):
        """Check if the error reduction from splitting is sufficient."""
        if parent_sse == 0:
            return False
        improvement = (parent_sse - child_sse) / parent_sse
        return improvement >= self.error_threshold

    def _fit(self, y, exog=None):
        # store last `lag` for each series
        self._last_window = y[:, -self.lag :].copy()

        X, y_embedded = self._create_input_matrix(y)
        if X is None:
            # If not enough data, fit a single model on what's available
            self.tree_ = {"is_leaf": True, "node_id": 0}
            self.leaf_models_[0] = self._fit_pr_model(y[:, :-1], y[:, -1])
            return self

        current_alpha = self.significance

        # Initialize the tree with a root node
        self.tree_ = {0: {"X": X, "y": y_embedded, "is_leaf": False}}
        node_queue = [0]
        next_node_id = 1

        for _depth in range(self.max_depth):
            if not node_queue:
                break

            nodes_at_this_level = list(node_queue)
            node_queue.clear()

            for node_id in nodes_at_this_level:
                node = self.tree_[node_id]

                # Try to find the best split for the current node
                best_split = self._find_optimal_split(node["X"], node["y"])

                if best_split["sse"] == float("inf"):
                    node["is_leaf"] = True
                    continue

                # --- Stopping Criteria ---
                parent_model = self._fit_pr_model(node["X"], node["y"])
                parent_sse = self._calculate_sse(parent_model, node["X"], node["y"])
                child_sse = best_split["sse"]

                is_good_split = False
                if self.stopping_criteria == "lin_test":
                    is_good_split = self._check_linearity(
                        parent_sse, child_sse, len(node["y"]), current_alpha
                    )
                elif self.stopping_criteria == "error_imp":
                    is_good_split = self._check_error_improvement(parent_sse, child_sse)
                elif self.stopping_criteria == "both":
                    is_good_split = self._check_linearity(
                        parent_sse, child_sse, len(node["y"]), current_alpha
                    ) and self._check_error_improvement(parent_sse, child_sse)

                if is_good_split:
                    node["split_info"] = {
                        "lag_idx": best_split["lag_idx"],
                        "threshold": best_split["threshold"],
                    }

                    # Create left child
                    left_id = next_node_id
                    self.tree_[left_id] = {
                        "X": node["X"][best_split["left_indices"]],
                        "y": node["y"][best_split["left_indices"]],
                        "is_leaf": False,
                    }
                    node["left_child"] = left_id
                    node_queue.append(left_id)
                    next_node_id += 1

                    # Create right child
                    right_id = next_node_id
                    self.tree_[right_id] = {
                        "X": node["X"][best_split["right_indices"]],
                        "y": node["y"][best_split["right_indices"]],
                        "is_leaf": False,
                    }
                    node["right_child"] = right_id
                    node_queue.append(right_id)
                    next_node_id += 1
                else:
                    node["is_leaf"] = True

            # Decrease alpha for the next level
            current_alpha /= self.significance_divider

        # Fit models for all leaf nodes
        for node_id, node in self.tree_.items():
            if node["is_leaf"] or node.get("left_child") is None:
                self.leaf_models_[node_id] = self._fit_pr_model(node["X"], node["y"])
                node["is_leaf"] = True

        return self

    def _predict(self, y=None, exog=None):
        self._check_is_fitted()

        # If y is not provided, we predict from the end of the training data
        if y is None:
            history = self._last_window.flatten()
        else:
            # Ensure y is 2D
            if y.ndim == 1:
                y = y.reshape(1, -1)
            # We take the last 'lag' points of y for the initial prediction
            history = y[0, -self.lag :]

        predictions = []

        for _ in range(self.horizon):
            # Find the leaf node for the current history
            current_node_id = 0
            while not self.tree_[current_node_id]["is_leaf"]:
                node = self.tree_[current_node_id]
                split_info = node["split_info"]

                # Lags are flipped to match the paper's L1, L2... convention
                lag_val = np.flip(history)[split_info["lag_idx"]]

                if lag_val < split_info["threshold"]:
                    current_node_id = node["left_child"]
                else:
                    current_node_id = node["right_child"]

            # Predict using the leaf's model
            leaf_model = self.leaf_models_[current_node_id]
            next_pred = leaf_model.predict(history.reshape(1, -1))[0]
            predictions.append(next_pred)

            # Update history for the next prediction
            history = np.append(history[1:], next_pred)

        return np.array(predictions[self.horizon - 1])
