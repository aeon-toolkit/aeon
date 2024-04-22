import pandas as pd

__all__ = [
    "TYPE_REGISTER_PROBA",
    "TYPE_LIST_PROBA",
]


TYPE_REGISTER_PROBA = [
    ("pred_interval", "Proba", "predictive intervals"),
    ("pred_quantiles", "Proba", "quantile predictions"),
    ("pred_var", "Proba", "variance predictions"),
]

TYPE_LIST_PROBA = pd.DataFrame(TYPE_REGISTER_PROBA)[0].values
