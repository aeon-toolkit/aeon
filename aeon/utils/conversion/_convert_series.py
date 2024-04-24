"""Series data converters.

This contains all functions to convert supported collection data types.

String identifier meanings (from aeon.utils.conversion import COLLECTIONS_DATA_TYPES) :
np.ndarray : 1 or 2D numpy array of time series
pd.Series:
pd.DataFrame:
"""

import numpy as np
import pandas as pd


def _resolve_input_type(y):
    """Resolve the input type of y.

    Parameters
    ----------
    y : np.ndarray, pd.Series, pd.DataFrame
        Time series to be converted.

    Returns
    -------
    str
        String identifier of the input type.
    """
    if isinstance(y, pd.Series):
        return "pd.Series"
    if isinstance(y, pd.DataFrame):
        return "pd.DataFrame"
    if isinstance(y, np.ndarray):
        return "np.ndarray"
    raise ValueError(f"Unknown input type: {type(y)}")


def _resolve_output_type(output_type):
    """Resolve the output type of y.

    Parameters
    ----------
    output_type : str or list of str
        The type to convert y to. If a list of string, either the type of y is in the
        list, in which case y is returned, or conversion is attempted on the first
        entry in the list of a valid output_type ["np.ndarray", "pd.Series",
        "pd.DataFrame"].

    Returns
    -------
    str
        String identifier of the output type.
    """
    valid_types = ["np.ndarray", "pd.Series", "pd.DataFrame"]
    # Check if output_type is a list
    if isinstance(output_type, list):
        # Iterate over the list and find the first valid type
        for typ in output_type:
            if typ in valid_types:
                return typ
        # If none in the list are valid, raise an error
        raise ValueError("None of the types in the list are valid output types")
    if isinstance(output_type, str) and output_type in valid_types:
        return output_type
    raise ValueError(f"Unknown output type {output_type}")


def convert_series(y, output_type):
    """Convert series y to the specified output_type.

    y, a single series of type pd.Series, pd.DataFrame or np.ndarray.

    Parameters
    ----------
    y : np.ndarray, pd.Series, pd.DataFrame
        Time series to be converted.
    output_type : string or list of string
        The type to convert y to. If a list of string, either the type of y is in the
        list, in which case y is returned, or conversion is attempted on the first
        entry in the list of a valid output_type ["np.ndarray", "pd.Series",
        "pd.DataFrame"].

    Returns
    -------
    converted version of y
    """
    input_type = _resolve_input_type(y)
    # If input type in the list, return it
    if isinstance(output_type, list):
        if input_type in output_type:
            return y
    elif isinstance(output_type, str):
        if input_type == output_type:
            return y
    else:
        raise ValueError("Unknown output type must be list or string")
    output_type = _resolve_output_type(output_type)
    if input_type == output_type:
        return y
    if output_type == "np.ndarray":
        return y.to_numpy()
    if output_type == "pd.Series":
        if input_type == "pd.DataFrame":
            if y.shape == (1, 1):  # special case of single element, cant squeeze
                y = y[y.columns[0]]
            else:
                y = y.squeeze()
            if y.ndim > 1:
                raise ValueError(
                    "pd.DataFrame of more than one row or column, cannot convert to "
                    "pd.Series"
                )
            return y
        elif input_type == "np.ndarray":
            y = y.squeeze()
            if y.ndim > 1:
                raise ValueError(
                    "np.ndarray of more than one row or column, cannot convert to "
                    "pd.Series"
                )
            return pd.Series(y)
    if output_type == "pd.DataFrame":
        if input_type == "pd.Series":
            return y.to_frame()
        elif input_type == "np.ndarray":
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
