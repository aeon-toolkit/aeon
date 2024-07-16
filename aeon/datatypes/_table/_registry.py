import pandas as pd

__all__ = [
    "TYPE_REGISTER_TABLE",
    "TYPE_LIST_TABLE",
]


TYPE_REGISTER_TABLE = [
    ("pd_DataFrame_Table", "Table", "pd.DataFrame representation of a data table"),
    ("numpy1D", "Table", "1D np.narray representation of a univariate table"),
    ("numpy_Table", "Table", "2D np.narray representation of a univariate table"),
    ("pd_Series_Table", "Table", "pd.Series representation of a data table"),
    ("list_of_dict", "Table", "list of dictionaries with primitive entries"),
]

TYPE_LIST_TABLE = pd.DataFrame(TYPE_REGISTER_TABLE)[0].values
