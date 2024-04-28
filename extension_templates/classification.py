"""
Extension template for time series classifiers.

Purpose of this implementation template:
    quick implementation of new estimators following the template. This is NOT a
    concrete class to import nor is it a base class to inherit from. This is to be used
    as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory abstract methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, classes_,
    n_classes_, fit_time_, _class_dictionary, _n_jobs, _tags, _tags_dynamic
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to aeon via PR

Mandatory components:
    fitting                 - _fit(self, X, y)
    predicting classes      - _predict(self, X)

Optional components:
    data conversion and capabilities tags - _tags
    fitted parameter inspection           - _get_fitted_params()
    predicting class probabilities        - _predict_proba(self, X)

Testing - implement if aeon classifier (not needed locally):
    get default parameters for test instance(s) - get_test_params()

"""

import numpy as np

from aeon.classification.base import BaseClassifier

# todo: add any necessary imports here


class MyTimeSeriesClassifier(BaseClassifier):
    """Custom time series classifier. todo: write docstring.

    todo: describe your custom time series classifier here

    Parameters
    ----------
    param_a : int
        descriptive explanation of param_a.
    param_b : string, default='default'
        descriptive explanation of param_b.
    param_c : boolean, default= True
        descriptive explanation of param_c.

    Attributes
    ----------
    att_a_ : int
        attributes are not passed to the constructor. If set in fit use following
        underscore.

    See Also
    --------
    Optional, related classes

    Notes
    -----
    Optional, Anything you want

    References
    ----------
    Optional, details of paper where algorithm described


    Examples
    --------
    todo: Add a simple use case example here
    """

    # todo: if any imports are aeon soft dependencies:
    #  * make sure to fill in the "python_dependencies" tag with the package import name

    # optional todo: override base class estimator default tags here if necessary
    # these are the default values, only add if different to these.
    _tags = {
        "X_inner_type": "numpy3D",  # which type do _fit/_predict accept, usually
        # this is usually "numpy3D" for equal length time series, np-list for unequal
        # length time series, see datatypes/panel/_registry.py for options.
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    # todo: add any parameters to constructor
    def __init__(self, param_a, param_b="default", param_c=None):
        # Note that parameters passed and set in constructor should not be changed
        # anywhere else. This is in order to comply with scikit learn structure.
        # Instead, copy into local variables in fit and predict and use these.

        # todo: copy parameters to self, use same names
        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c

        # todo: change "MyTimeSeriesClassifier" to the name of the class
        super().__init__()

    # todo: implement this abstract function
    def _fit(self, X, y):
        """Fit time series classifier to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_type")
            if self.get_tag("X_inner_type") = "numpy3D":
                3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
        y : 1D np.array of int, of shape [n_cases] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.
        """

        # implement here
        # IMPORTANT: avoid side effects to X, y, if changing X and y make local copies.

    # todo: implement this abstract function
    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_type")
            if self.get_tag("X_inner_type") = "numpy3D":
                3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            if self.get_tag("X_inner_type") = "np-list":
                list of 2D np.ndarray of shape = (n_cases,)

        Returns
        -------
        y : 1D np.array of int, of shape [n_cases] - predicted class labels
            indices correspond to instance indices in X
        """

    # todo: consider implementing this, optional
    # if you do not implement it, then the default _predict_proba will be called.
    # the default simply calls predict and sets probas to 0 or 1.

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_type")
            if self.get_tag("X_inner_type") = "numpy3D":
                3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            if self.get_tag("X_inner_type") = "np-list":
                list of 2D np.ndarray of shape = (n_cases,)

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """

        # implement here
        # IMPORTANT: avoid side effects to X

    # todo: consider implementing this, optional
    # implement only if different from default:
    #   default retrieves all self attributes ending in "_"
    #   and returns them with keys that have the "_" removed
    # if not implementing, delete the method
    #   avoid overriding get_fitted_params
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        # implement here
        #
        # when this function is reached, it is already guaranteed that self is fitted
        #   this does not need to be checked separately
        #
        # parameters of components should follow the sklearn convention:
        #   separate component name from parameter name by double-underscore
        #   e.g., componentname__paramname

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contacting
                    functionality
                "train_estimate" - used in some classifiers that set the
                    "capability:train_estimate" tag to True to allow for more efficient
                    testing when relevant parameters are available

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        return {"param_a": 42, "param_c": False}
