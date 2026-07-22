"""Random over-sampling for imbalanced collections of time series."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["RandomOverSampler"]

from collections import OrderedDict

import numpy as np
from sklearn.utils import check_random_state

from aeon.transformations.collection import BaseCollectionTransformer


class RandomOverSampler(BaseCollectionTransformer):
    """Random over-sampling for imbalanced collections of time series.

    Replicates samples from each non-majority class with replacement until every
    class has the same number of cases as the majority class. Original samples
    are kept; only synthetic duplicates are appended.

    This mirrors ``imblearn.over_sampling.RandomOverSampler`` with
    ``sampling_strategy="all"`` for equal-length univariate collections.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Controls the random number generation used when drawing replacement
        samples.

    Examples
    --------
    >>> from aeon.transformations.collection.imbalance import RandomOverSampler
    >>> import numpy as np
    >>> X = np.random.randn(10, 1, 12)
    >>> y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    >>> X_res, y_res = RandomOverSampler(random_state=0).fit_transform(X, y)
    >>> y_res.shape[0] == 14
    True
    """

    _tags = {
        "requires_y": True,
    }

    def __init__(self, random_state=None):
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        self._random_state = check_random_state(self.random_state)
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        return self

    def _transform(self, X, y=None):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            choose = self._random_state.choice(
                target_class_indices, size=n_samples, replace=True
            )
            X_resampled.append(X[choose])
            y_resampled.append(y[choose])

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"random_state": 0}
