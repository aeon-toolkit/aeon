import pandas as pd
from deprecated.sphinx import deprecated


# TODO: remove in v0.8.0
@deprecated(
    version="0.6.0",
    reason="This function has moved to utils.conversion and this version will be "
    "deleted in V0.8.0.",
    category=FutureWarning,
)
def is_nested_univ_dataframe(X):
    """Check if X is nested dataframe."""
    # Otherwise check all entries are pd.Series
    if not isinstance(X, pd.DataFrame):
        return False
    for _, series in X.items():
        for cell in series:
            if not isinstance(cell, pd.Series):
                return False
    return True
