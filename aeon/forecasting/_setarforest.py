"""SETAR-Forest: An ensemble of SETAR-Trees for global time series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeon.forecasting.base import BaseForecaster

from ._setartree import SETARTree

__maintainer__ = ["TinaJin0228"]
__all__ = ["SETARForest"]


class SETARForest(BaseForecaster):
    """
    SETAR-Forest: Bagging + random subspace ensemble of SETAR-Tree base learners.

    This implementation is based on the codebase associated with the work
    by Godahewa et al. [1].

    In a SETAR-Forest, each tree is trained on a bootstrap-free row sample of the
    embedded matrix and a random subset of features (lags, and in future, covariates).
    Predictions are averaged across trees.

    The forest does not build trees itself. It reuses `SETARTree` for all splitting,
    linearity/error tests, and leaf model training, via `SETARTree.fit_from_embedded`.

    Parameters
    ----------
    lag : int, default=10
        Number of past lags used as features (must match the base tree setting).
    n_estimators : int, default=10
        Number of trees (R: `bagging_freq`).
    bagging_fraction : float in (0,1], default=0.8
        Fraction of embedded rows for each tree (R: `bagging_fraction`).
    feature_fraction : float in (0,1], default=1.0
        Fraction of features (columns, excluding `y`) for each tree.
    max_depth : int, default=1000
        Max depth passed to base trees.
    stopping_criteria : {"lin_test", "error_imp", "both"}, default="both"
        Stopping rule passed to base trees.
    significance : float, default=0.05
        Initial alpha for base trees
        (overridden per-tree if `random_tree_significance`).
    seq_significance : bool, default=True
        Decrease alpha by `significance_divider` per level in base trees.
    significance_divider : int, default=2
        Divider for alpha per depth level.
    error_threshold : float, default=0.03
        Minimum error reduction for a split
        (overridden if `random_tree_error_threshold`).
    n_thresholds : int, default=15
        Number of candidate cut points per lag in base trees.
    scale : bool, default=False
        If True, scale series by mean before embedding
        (delegated to base tree for predict).
    random_tree_significance : bool, default=False
        If True, choose a random alpha in [0.01, 0.1] per tree (uniform grid, as in R).
    random_significance_divider : bool, default=False
        If True, choose a random integer divider in {2,...,10} per tree.
    random_tree_error_threshold : bool, default=False
        If True, uniformly choose a random error threshold in [0.001, 0.05] per tree.
    integer_conversion : bool, default=False
        If True, round the *final averaged* prediction to nearest integer.
    random_state : Optional[int], default=None
        RNG seed for reproducibility.

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
        lag: int = 10,
        n_estimators: int = 10,
        bagging_fraction: float = 0.8,
        feature_fraction: float = 1.0,
        max_depth: int = 1000,
        stopping_criteria: str = "both",
        significance: float = 0.05,
        seq_significance: bool = True,
        significance_divider: int = 2,
        error_threshold: float = 0.03,
        n_thresholds: int = 15,
        scale: bool = False,
        random_tree_significance: bool = False,
        random_significance_divider: bool = False,
        random_tree_error_threshold: bool = False,
        integer_conversion: bool = False,
        random_state: int | None = None,
    ):
        super().__init__(horizon=1, axis=0)
        self.lag = lag
        self.n_estimators = n_estimators
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.max_depth = max_depth
        self.stopping_criteria = stopping_criteria
        self.significance = significance
        self.seq_significance = seq_significance
        self.significance_divider = significance_divider
        self.error_threshold = error_threshold
        self.n_thresholds = n_thresholds
        self.scale = scale
        self.random_tree_significance = random_tree_significance
        self.random_significance_divider = random_significance_divider
        self.random_tree_error_threshold = random_tree_error_threshold
        self.integer_conversion = integer_conversion
        self.random_state = random_state

        # learned attributes
        self.estimators_: list[SETARTree] = []
        self.feature_subsets_: list[list[str]] = []
        self.embedded_columns_: list[str] | None = None  # for reference/debug

    def _rng(self, offset: int = 0) -> np.random.Generator:
        seed = None if self.random_state is None else (self.random_state + offset)
        return np.random.default_rng(seed)

    def _embed_once(self, y, exog=None) -> pd.DataFrame:
        """Use a template SETARTree to build the full embedded matrix once."""
        template = SETARTree(
            lag=self.lag,
            max_depth=1,  # depth irrelevant for embedding
            stopping_criteria=self.stopping_criteria,
            significance=self.significance,
            significance_divider=self.significance_divider,
            error_threshold=self.error_threshold,
            n_thresholds=self.n_thresholds,
            scale=self.scale,
            seq_significance=self.seq_significance,
        )
        training_set = template._process_input_data(y, exog)
        embedded_df, _, _ = template._create_tree_input_matrix(training_set)
        return embedded_df

    def _fit(self, y, exog=None):
        # Full embedded matrix (reuses SETARTree logic)
        embedded_df = self._embed_once(y, exog)
        if embedded_df.empty:
            raise ValueError("SETARForest: empty embedded matrix; insufficient data.")
        self.embedded_columns_ = list(embedded_df.columns)

        # columns: ["y", "Lag1", ... "LagL", (future: covariates...)]
        all_features = [c for c in embedded_df.columns if c != "y"]
        n_rows = embedded_df.shape[0]
        n_feats = len(all_features)

        n_rows_bag = max(1, int(round(self.bagging_fraction * n_rows)))
        n_feats_bag = max(1, int(round(self.feature_fraction * n_feats)))

        self.estimators_.clear()
        self.feature_subsets_.clear()

        # Train n_estimators trees on bagged rows/features
        for b in range(self.n_estimators):
            rng = self._rng(b + 1)

            # row bagging
            row_idx = np.sort(rng.choice(n_rows, size=n_rows_bag, replace=False))

            # feature subsampling
            feat_idx = np.sort(rng.choice(n_feats, size=n_feats_bag, replace=False))
            feat_subset = [all_features[i] for i in feat_idx]

            # per-tree randomized hyperparameters
            significance = self.significance
            sig_div = self.significance_divider
            err_thr = self.error_threshold

            if self.random_tree_significance:
                # sample seq(0.01, 0.1, length.out = 10) -> 10 uniform grid points
                grid = np.linspace(0.01, 0.10, num=10)
                significance = rng.choice(grid)

            if self.random_significance_divider:
                sig_div = int(rng.integers(2, 11))  # {2,...,10}

            if self.random_tree_error_threshold:
                # seq(0.001, 0.05, length.out = 50)
                grid = np.linspace(0.001, 0.05, num=50)
                err_thr = rng.choice(grid)

            # prepare current bagged design
            bag_df = embedded_df.iloc[row_idx, :]
            bag_df = bag_df[["y"] + feat_subset]

            # create and fit the base tree via tree's API
            tree = SETARTree(
                lag=self.lag,
                max_depth=self.max_depth,
                stopping_criteria=self.stopping_criteria,
                significance=significance,
                significance_divider=sig_div,
                error_threshold=err_thr,
                n_thresholds=self.n_thresholds,
                scale=self.scale,
                seq_significance=self.seq_significance,
                feature_subset=feat_subset,  # restrict split candidates
            ).fit_from_embedded(bag_df)

            self.estimators_.append(tree)
            self.feature_subsets_.append(feat_subset)

        # self.is_fitted = True
        return self

    def _predict(self, y, exog=None):
        if not self.estimators_:
            raise RuntimeError("SETARForest not fitted.")
        preds = np.array(
            [est.predict(y, exog) for est in self.estimators_], dtype=float
        )
        avg = float(np.mean(preds))
        if self.integer_conversion:
            avg = float(np.rint(avg))
        return avg

    def _forecast(self, y, exog=None):
        """Fit-then-predict one step (aeon contract for horizon=1)."""
        self.fit(y, exog)
        self.forecast_ = self.predict(y, exog)
        return self.forecast_
