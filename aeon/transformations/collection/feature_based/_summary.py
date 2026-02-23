"""Summary feature transformer."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SevenNumberSummary"]

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.numba.stats import (
    row_mean,
    row_numba_max,
    row_numba_min,
    row_quantile,
    row_std,
)


class SevenNumberSummary(BaseCollectionTransformer):
    """Seven-number summary transformer.

    Transforms a time series into seven basic summary statistics.

    Parameters
    ----------
    summary_stats : ["default", "quantiles", "bowley", "tukey"], default="default"
        The summary statistics to compute.
        The options are as follows, with float denoting the percentile value extracted
        from the series:
            - "default": mean, std, min, max, 0.25, 0.5, 0.75
            - "quantiles": 0.0215, 0.0887, 0.25, 0.5, 0.75, 0.9113, 0.9785
            - "bowley": min, max, 0.1, 0.25, 0.5, 0.75, 0.9
            - "tukey": min, max, 0.125, 0.25, 0.5, 0.75, 0.875

    Examples
    --------
    >>> from aeon.transformations.collection.feature_based import SevenNumberSummary  # noqa
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X = make_example_3d_numpy(n_cases=4, n_channels=1, n_timepoints=10,
    ...                           random_state=0, return_y=False)
    >>> tnf = SevenNumberSummary()
    >>> tnf.fit(X)
    SevenNumberSummary(...)
    >>> print(tnf.transform(X)[0])
    [1.12176987 0.52340259 0.         1.92732552 0.8542758  1.14764656
     1.39573111]
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        summary_stats="default",
    ):
        self.summary_stats = summary_stats

        super().__init__()

    def _transform(self, X, y=None):
        n_cases = len(X)
        n_channels, _ = X[0].shape

        functions = self._get_functions()

        Xt = np.zeros((n_cases, 7 * n_channels))
        for i in range(n_cases):
            for n, f in enumerate(functions):
                idx = n * n_channels
                if isinstance(f, float):
                    Xt[i, idx : idx + n_channels] = row_quantile(X[i], f)
                else:
                    Xt[i, idx : idx + n_channels] = f(X[i])

        return Xt

    def _get_functions(self):
        if self.summary_stats == "default":
            return [
                row_mean,
                row_std,
                row_numba_min,
                row_numba_max,
                0.25,
                0.5,
                0.75,
            ]
        elif self.summary_stats == "quantiles":
            return [
                0.0215,
                0.0887,
                0.25,
                0.5,
                0.75,
                0.9113,
                0.9785,
            ]
        elif self.summary_stats == "bowley":
            return [
                row_numba_min,
                row_numba_max,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
            ]
        elif self.summary_stats == "tukey":
            return [
                row_numba_min,
                row_numba_max,
                0.125,
                0.25,
                0.5,
                0.75,
                0.875,
            ]
        else:
            raise ValueError(
                f"Summary function input {self.summary_stats} not recognised."
            )
