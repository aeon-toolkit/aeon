from typing import Optional, Union

import numpy as np
from numba.typed import List as NumbaList


def _is_numpy_list_multivariate(
    x: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
) -> bool:
    if y is None:
        if isinstance(x, np.ndarray):
            x_dims = x.ndim
            if x_dims == 3:
                if x.shape[1] == 1:
                    return False
                return True
            if x_dims == 2:
                # As this function is used for pairwise we assume it isnt a single
                # multivariate time series but two collections of univariate
                return False
            if x_dims == 1:
                return False

        if isinstance(x, (list, NumbaList)):
            x_dims = x[0].ndim
            if x_dims == 2:
                if x[0].shape[0] == 1:
                    return False
                return True
            if x_dims == 1:
                return False
    else:
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_dims = x.ndim
            y_dims = y.ndim
            if x_dims < y_dims:
                return _is_numpy_list_multivariate(y, x)

            if x_dims == 3 and y_dims == 3:
                if x.shape[1] == 1 and y.shape[1] == 1:
                    return False
                return True
            if x_dims == 3 and y_dims == 2:
                if x.shape[1] == 1:
                    return False
                return True
            if x_dims == 3 and y_dims == 1:
                if x.shape[1] == 1:
                    return False
            if x_dims == 2 and y_dims == 2:
                # If two 2d arrays passed as this function is used for pairwise we
                # assume it isnt two multivariate time series but two collections of
                # univariate
                return False
            if x_dims == 2 and y_dims == 1:
                return False
            if x_dims == 1 and y_dims == 1:
                return False
        if isinstance(x, (list, NumbaList)) and isinstance(y, (list, NumbaList)):
            x_dims = x[0].ndim
            y_dims = y[0].ndim

            if x_dims < y_dims:
                return _is_numpy_list_multivariate(y, x)

            if x_dims == 1 or y_dims == 1:
                return False

            if x_dims == 2 and y_dims == 2:
                if x[0].shape[0] == 1 or y[0].shape[0] == 1:
                    return False
                return True
        list_x = None
        ndarray_y: Optional[np.ndarray] = None
        if isinstance(x, (list, NumbaList)):
            list_x = x
            ndarray_y = y
        elif isinstance(y, (list, NumbaList)):
            list_x = y
            ndarray_y = x

        if list_x is not None and ndarray_y is not None:
            list_y = []
            if ndarray_y.ndim == 3:
                for i in range(ndarray_y.shape[0]):
                    list_y.append(ndarray_y[i])
            else:
                list_y = [ndarray_y]
            return _is_numpy_list_multivariate(list_x, list_y)

    raise ValueError("The format of you input is not supported.")


def convert_collection_to_numba_list(
    x: Union[np.ndarray, list[np.ndarray]],
    name: str = "X",
    multivariate_conversion: bool = False,
) -> NumbaList[np.ndarray]:
    """Convert input collections to a list of arrays for pairwise distance calculation.

    Takes a single or multiple time series and converts them to a list of 2D arrays. If
    the input is a single time series, it is reshaped to a 2D array as the sole element
    of a list. If the input is a 2D array of shape (n_cases, n_timepoints), it is
    reshaped to a list of n_cases 1D arrays with n_timepoints points. A 3D array is
    converted to a list with n_cases 2D arrays of shape (n_channels, n_timepoints).
    Lists of 1D arrays are converted to lists of 2D arrays.

    Parameters
    ----------
    x : Union[np.ndarray, List[np.ndarray]]
        One or more time series of shape (n_cases, n_channels, n_timepoints) or
        (n_cases, n_timepoints) or (n_timepoints,).
    name : str, optional
        Name of the variable to be converted for error handling, by default "X".
    multivariate_conversion : bool, optional
        Boolean indicating if the conversion should be multivariate, by default False.
        If True, the input is assumed to be multivariate and reshaped accordingly.
        If False, the input is reshaped to univariate.

    Returns
    -------
    NumbaList[np.ndarray]
        Numba typedList of 2D arrays with shape (n_channels, n_timepoints) of length
        n_cases.
    bool
        Boolean indicating if the time series is of unequal length. True if the time
        series are of unequal length, False otherwise.

    Raises
    ------
    ValueError
        If x is not a 1D, 2D or 3D array or a list of 1D or 2D arrays.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return NumbaList(x), False
        elif x.ndim == 2:
            if multivariate_conversion:
                return NumbaList(x.reshape(1, x.shape[0], x.shape[1])), False
            return NumbaList(x.reshape(x.shape[0], 1, x.shape[1])), False
        elif x.ndim == 1:
            return NumbaList(x.reshape(1, 1, x.shape[0])), False
        else:
            raise ValueError(f"{name} must be 1D, 2D or 3D")
    elif isinstance(x, (list, NumbaList)):
        x_new = NumbaList()
        expected_n_timepoints = x[0].shape[-1]
        unequal_timepoints = False
        for i in range(len(x)):
            curr_x = x[i]
            if curr_x.shape[-1] != expected_n_timepoints:
                unequal_timepoints = True
            if x[i].ndim == 2:
                x_new.append(curr_x)
            elif x[i].ndim == 1:
                x_new.append(curr_x.reshape((1, curr_x.shape[0])))
            else:
                raise ValueError(f"{name} must include only 1D or 2D arrays")
        return x_new, unequal_timepoints
    else:
        raise ValueError(f"{name} must be either np.ndarray or List[np.ndarray]")
