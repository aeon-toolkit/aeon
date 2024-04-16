"""Series data converters.

This contains all functions to convert supported collection data types.

String identifier meanings (from aeon.utils.conversion import COLLECTIONS_DATA_TYPES) :
np.ndarray : 1 or 2D numpy array of time series
pd.Series:
pd.DataFrame:
"""

import numpy as np
import pandas as pd


def convert_series(y, output_type):
    """Convert series y to the specified output_type.

    y, a single series of type pd.Series, pd.DataFrame or np.ndarray.

    Parameters
    ----------
    y : np.ndarray, pd.Series, pd.DataFrame
        Time series to be converted
    output_type : string
        The type to convert y to

    Returns
    -------
    converted version of y
    """
    if output_type == "np.ndarray":
        if isinstance(y, (pd.Series, pd.DataFrame)):
            return y.to_numpy()
        elif isinstance(y, np.ndarray):
            return y
        else:
            raise ValueError(f"Cannot convert: {type(y)} to np.ndarray")
    elif output_type == "pd.Series":
        if isinstance(y, pd.Series):
            return y
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError(
                    "DataFrame more than one column, cannot convert to Series"
                )
            y = y.iloc[:, 0]
            return y
        elif isinstance(y, np.ndarray):
            y = y.squeeze()
            if y.ndim == 1:
                # Convert 1D numpy array to pandas Series
                return pd.Series(y)
            else:
                raise ValueError(
                    "Cannot convert array with more than one dimension to a pd.Series"
                )
    elif output_type == "pd.DataFrame":
        if isinstance(y, pd.DataFrame):
            return y
        elif isinstance(y, pd.Series):
            return y.to_frame()
        elif isinstance(y, np.ndarray):
            return pd.DataFrame(y)
        else:
            raise ValueError(
                f"Unknown output_type: {output_type} must be one of "
                f"pd.Series, pd.DataFrame or np.ndarray"
            )
    else:
        raise ValueError(
            f"Unknown output_type: {output_type} must be one of "
            f"pd.Series, pd.DataFrame or np.ndarray"
        )
