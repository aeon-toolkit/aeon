"""Common functions for unequal length transformations.

These should ideally be incorporated into the collection data utilities in utils/ in
the future.
"""

import numpy as np


def _get_min_length(X):
    if isinstance(X, np.ndarray):
        return X.shape[2]
    else:
        return min([x.shape[1] for x in X])


def _get_max_length(X):
    if isinstance(X, np.ndarray):
        return X.shape[2]
    else:
        return max([x.shape[1] for x in X])
