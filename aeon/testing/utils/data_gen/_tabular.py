"""Tabular generators for testing."""

import pandas as pd
from sklearn.utils import check_random_state


def _make_primitives(n_columns=1, random_state=None):
    """Generate one or more primitives, for checking inverse-transform."""
    rng = check_random_state(random_state)
    if n_columns == 1:
        return rng.rand()
    return rng.rand(size=(n_columns,))


def _make_tabular_X(n_instances=20, n_columns=1, return_numpy=True, random_state=None):
    """Generate tabular X, for checking inverse-transform."""
    rng = check_random_state(random_state)
    X = rng.rand(n_instances, n_columns)
    if return_numpy:
        return X
    else:
        return pd.DataFrame(X)
