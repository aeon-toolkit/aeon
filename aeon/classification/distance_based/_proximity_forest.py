"""Generate candidate splitter."""

import numpy as np

# from numba import njit

# from sklearn.preprocessing import LabelEncoder


# @njit(cache=True, fastmath=True)
def get_parameter_value(X=None):
    """Generate random parameter values.

    For a list of distance measures, generate a dictionary
    of parameterized distances.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_timepoints)

    Returns
    -------
    distance_param : a list of distances and their
        parameters.
    """
    X_std = X.std()
    param_ranges = {
        "euclidean": {},
        "dtw": {"window": (0, 0.25)},
        "ddtw": {"window": (0, 0.25)},
        "wdtw": {"g": (0, 1)},
        "wddtw": {"g": (0, 1)},
        "erp": {"g": (X_std / 5, X_std)},
        "lcss": {"epsilon": (X_std / 5, X_std), "window": (0, 0.25)},
    }
    random_params = []
    for measure, ranges in param_ranges.items():
        random_params.append(
            {
                measure: {
                    param: np.round(np.random.uniform(low, high), 3)
                    for param, (low, high) in ranges.items()
                }
            }
        )

    # For TWE
    lmbda = np.random.randint(0, 9)
    exponent_range = np.arange(1, 6)  # Exponents from -5 to 1 (inclusive)
    random_exponent = np.random.choice(exponent_range)
    nu = 1 / 10**random_exponent
    random_params.append({"twe": {"lmbda": lmbda, "nu": nu}})

    # For MSM
    base = 10
    # Exponents from -2 to 2 (inclusive)
    exponents = np.arange(-2, 3, dtype=np.float64)

    # Randomly select an index from the exponent range
    random_index = np.random.randint(0, len(exponents))
    c = base ** exponents[random_index]
    random_params.append({"msm": {"c": c}})

    return random_params


# @njit(cache=True, fastmath=True)
def get_candidate_splitter(X, y, paramterized_distances):
    """Generate candidate splitter.

    Takes a time series dataset and a set of parameterized
    distance measures to create a candidate splitter, which
    contains a parameterized distance measure and a set of exemplars.

    Parameters
    ----------
    X : np.ndarray shape (n_cases, n_timepoints)
        The training input samples.
    y : np.array shape (n_cases,) or (n_cases,1)
    parameterized_distances : list
        Contains the distances and their parameters.

    Returns
    -------
    splitter : list of two dictionaries
        A distance and its parameter values and a set of exemplars.
    """
    _X = X
    _y = y
    # label_encoder = LabelEncoder()
    # _y_label = label_encoder.fit_transform(_y)
    # if _y_label.ndim == 1:
    #    _y_label = _y_label.reshape(-1,1)
    # _X_y = np.concatenate([_X,_y_label], axis=1)

    # Now, I need to create a dictionary
    # where the keys will be unique classes and values will be a random
    # data of that class
    exemplars = []
    for label in np.unique(_y):
        y_new = _y[_y == label]
        X_new = _X[_y == label]
        id = np.random.randint(0, X_new.shape[0])
        exemplars.append({y_new[id]: X_new[id, :]})

    # Create a list with first element exemplars and second element a random
    # parameterized distance measure
    n = np.random.randint(0, 9)
    splitter = [exemplars, paramterized_distances[n]]

    return splitter
