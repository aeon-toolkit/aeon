"""Register of estimator tags.

Note for extenders: new tags should be entered in ESTIMATOR_TAG_REGISTER.
No other place is necessary to add new tags.

This module exports the following:

---
ESTIMATOR_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - identifier for the base class of objects this tag applies to
                 must be in _base_classes.BASE_CLASS_IDENTIFIER_LIST
    2 : string - expected type of the tag value
        should be one of:
            "bool" - valid values are True/False
            "int" - valid values are all integers
            "str" - valid values are all strings
            "list" - valid values are all lists of arbitrary elements
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
            ("list", "str") - any individual string or list of strings is valid
        validity can be checked by check_tag_is_valid (see below)
    3 : string - plain English description of the tag
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ESTIMATOR_TAGS"]


ESTIMATOR_TAGS = {
    # all estimators
    "python_version": {
        "class": "estimator",
        "type": ["str", None],
        "description": "Python version specifier (PEP 440) for estimator as str. "
        "e.g. '>=3.6', '>=3.7, <3.9'. If None, no restriction.",
    },
    "python_dependencies": {
        "class": "estimator",
        "type": ["list", "str", None],
        "description": "Python dependency version specifier (PEP 440) for estimator "
        "as str or list of str. e.g. 'tsfresh>=0.20.0', "
        "'[tsfresh>=0.20.0, pandas<2.0.0]'. If None, no restriction.",
    },
    "non-deterministic": {
        "class": "estimator",
        "type": "bool",
        "description": "The estimator is non-deterministic, and multiple runs will "
        "not produce the same output even if a `random_state` is set.",
    },
    "cant-pickle": {
        "class": "estimator",
        "type": "bool",
        "description": "The estimator cannot be pickled.",
    },
    "X_inner_type": {
        "class": "estimator",
        "type": ("list", ["pd.DataFrame", "np.ndarray", "numpy3D", "np-list"]),
        "description": "What data structure(s) the estimator uses internally for "
        "fit/predict.",
    },
    "algorithm_type": {
        "class": "estimator",
        "type": (
            "str",
            [
                "dictionary",
                "distance",
                "feature",
                "hybrid",
                "interval",
                "convolution",
                "shapelet",
                "deeplearning",
            ],
        ),
        "description": "Which type the estimator falls under in its relevant taxonomy"
        " of time series machine learning algorithms.",
    },
    "fit_is_empty": {
        "class": "estimator",
        "type": "bool",
        "description": "fit contains no logic and can be skipped? Yes=True, No=False",
    },
    # capabilities - flags for what an estimator can and cannot do
    "capability:univariate": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator deal with single series input?",
    },
    "capability:multivariate": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator deal with series with multiple channels?",
    },
    "capability:missing_values": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator handle missing data (NA, np.nan) in inputs?",
    },
    "capability:unequal_length": {
        "class": "collection-estimator",
        "type": "bool",
        "description": "Can the estimator handle unequal length time series input?",
    },
    "capability:multithreading": {
        "class": "estimator",
        "type": "bool",
        "description": "Can the estimator set `n_jobs` to use multiple threads?",
    },
    "capability:train_estimate": {
        "class": ["classifier", "regressor"],
        "type": "bool",
        "description": "Can the estimator use an in-built mechanism to estimate its "
        "performance on the training set?",
    },
    "capability:contractable": {
        "class": ["classifier", "regressor"],
        "type": "bool",
        "description": "Can the estimator limiting max fit time?",
    },
    "capability:inverse_transform": {
        "class": "transformer",
        "type": "bool",
        "description": "Can the transformer carrying out an inverse transform?",
    },
    # other
    "returns_dense": {
        "class": "segmenter",
        "type": "bool",
        "description": "Does the segmenter return a list of change points/start index "
        "each segmenter (dense format) or a list indicating which segment each time "
        "point belongs to.",
    },
    "requires_y": {
        "class": "transformer",
        "type": "bool",
        "description": "Does this transformer require y to be passed in fit and "
        "transform?",
    },
    "capability:unequal_length:removes": {
        "class": "transformer",
        "type": "bool",
        "description": "Is the transformer result guaranteed to be equal length series "
        "or tabular?",
    },
    "capability:missing_values:removes": {
        "class": "transformer",
        "type": "bool",
        "description": "Is the transformer result guaranteed to have no missing "
        "values?",
    },
}
