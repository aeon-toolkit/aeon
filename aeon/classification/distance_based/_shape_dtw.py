"""ShapeDTW classifier.

Nearest neighbour classifier that extracts shape features.
"""

import numpy as np
from deprecated.sphinx import deprecated

# Tuning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import FunctionTransformer

# Classifiers
from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from aeon.transformations.collection.dictionary_based._paa import PAA
from aeon.transformations.collection.dwt import DWTTransformer

# Transformers
from aeon.transformations.collection.hog1d import HOG1DTransformer
from aeon.transformations.collection.segment import SlidingWindowSegmenter
from aeon.transformations.collection.slope import SlopeTransformer
from aeon.utils.numba.general import slope_derivative_3d

__maintainer__ = []


# TODO: remove in v0.9.0
@deprecated(
    version="0.8.0",
    reason="ShapeDTW classifier will be removed in v0.9.0.",
    category=FutureWarning,
)
class ShapeDTW(BaseClassifier):
    """
    ShapeDTW classifier.

    ShapeDTW [1]_ extracts a set of subseries describing local neighbourhoods around
    each data point in a time series. These subseries are then passed into a
    shape descriptor function that transforms these local neighbourhoods into a new
    representation. This new representation is then used for nearest neighbour
    classification with dynamic time warping.

    Parameters
    ----------
    n_neighbors : int, default =1
        Number of neighbours, k, for the k-NN classifier.
    subsequence_length : int, default=sqrt(n_timepoints)
        Length of the subseries to extract.
    shape_descriptor_function : str, default = 'raw'
        Defines the function to describe the set of subsequences
        The possible shape descriptor functions are as follows:
        - 'raw' : use the raw subsequence as the shape descriptor function.
        - 'paa' : use PAA as the shape descriptor function. params =
        num_intervals_paa (default=8).
        - 'dwt' : use DWT (Discrete Wavelet Transform) as the shape descriptor
        function. params = num_levels_dwt (default=3).
        - 'slope' : use the gradient of each subsequence fitted by a total least
        squares regression as the shape descriptor function. params =
        num_intervals_slope (default=8).
        - 'derivative' : use the derivative of each subsequence as the shape
        descriptor function.
        - 'hog1d' : use a histogram of gradients in one dimension as the shape
        descriptor function. params = num_intervals_hog1d (default=2), num_bins_hod1d
        (default=8), scaling_factor_hog1d (default=0.1).
        - 'compound'  : use a combination of two shape descriptors simultaneously.
        params = weighting_factor (default=None). Defines how to scale values of a
        shape descriptor.  If a value is not given, this value is tuned by 10-fold
        cross-validation on the training data.
    shape_descriptor_functions : List of str, default = ['raw','derivative']
        Only applicable when the shape_descriptor_function is set to 'compound'. Use
        a list of shape descriptor functions at the same time.
    metric_params : dict, default = None
        Dictionary for metric parameters.

    Notes
    -----
    .. [1] Jiaping Zhao and Laurent Itti, "shapeDTW: Shape Dynamic Time Warping",
        Pattern Recognition, 74, pp 171-184, 2018
        http://www.sciencedirect.com/science/article/pii/S0031320317303710,

    Example
    -----
    >>> from aeon.classification.distance_based import ShapeDTW
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = ShapeDTW()
    >>> clf.fit(X_train, y_train)
    ShapeDTW(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_neighbors=1,
        subsequence_length=30,
        shape_descriptor_function="raw",
        shape_descriptor_functions=None,
        metric_params=None,
    ):
        self.n_neighbors = n_neighbors
        self.subsequence_length = subsequence_length
        self.shape_descriptor_function = shape_descriptor_function
        self.shape_descriptor_functions = shape_descriptor_functions
        if shape_descriptor_functions is None:
            self._shape_descriptor_functions = ["raw", "derivative"]
        else:
            self._shape_descriptor_functions = shape_descriptor_functions
        self.metric_params = metric_params

        super().__init__()

    def _fit(self, X, y):
        """Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints)
        y : np.ndarray
            The training data class labels of shape (n_cases,).

        Returns
        -------
        self : the shapeDTW object
        """
        # Perform preprocessing on params.
        if not (isinstance(self.shape_descriptor_function, str)):
            raise TypeError(
                f"shape_descriptor_function must be an 'str'. Found "
                f"{type(self.shape_descriptor_function).__name__} instead."
            )
        if self.metric_params is None:
            self._metric_params = {}
        else:
            self._metric_params = self.metric_params

        # If the shape descriptor is 'compound',
        # calculate the appropriate weighting_factor
        if self.shape_descriptor_function == "compound":
            self._calculate_weighting_factor_value(X, y)

        # Fit the SlidingWindowSegmenter
        sw = SlidingWindowSegmenter(self.subsequence_length)
        sw.fit(X)
        self.sw = sw

        # Transform the training data.
        X = self._preprocess(X)

        # Fit the kNN classifier
        self.knn = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(X, y)
        self.classes_ = self.knn.classes_
        return self

    def _calculate_weighting_factor_value(self, X, y):
        """Calculate the appropriate weighting_factor.

        Check for the compound shape descriptor.
        If a value is given, the weighting_factor is set
        as the given value. If not, its tuned via
        a 10-fold cross-validation on the training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
        y - training data classes of shape [n_cases].
        """
        self._metric_params = {k.lower(): v for k, v in self._metric_params.items()}

        # Get the weighting_factor if one is provided
        if self._metric_params.get("weighting_factor") is not None:
            self.weighting_factor = self._metric_params.get("weighting_factor")
        else:
            # Tune it otherwise
            self._param_matrix = {
                "metric_params": [
                    {"weighting_factor": 0.1},
                    {"weighting_factor": 0.125},
                    {"weighting_factor": (1 / 6)},
                    {"weighting_factor": 0.25},
                    {"weighting_factor": 0.5},
                    {"weighting_factor": 1},
                    {"weighting_factor": 2},
                    {"weighting_factor": 4},
                    {"weighting_factor": 6},
                    {"weighting_factor": 8},
                    {"weighting_factor": 10},
                ]
            }

            n = self.n_neighbors
            sl = self.subsequence_length
            sdf = self.shape_descriptor_function
            sdfs = self._shape_descriptor_functions
            if sdfs is None or len(sdfs) != 2:
                raise ValueError(
                    "When using 'compound', "
                    + "shape_descriptor_functions must be a "
                    + "string array of length 2."
                )
            mp = self._metric_params

            grid = GridSearchCV(
                estimator=ShapeDTW(
                    n_neighbors=n,
                    subsequence_length=sl,
                    shape_descriptor_function=sdf,
                    shape_descriptor_functions=sdfs,
                    metric_params=mp,
                ),
                param_grid=self._param_matrix,
                cv=KFold(n_splits=10, shuffle=True),
                scoring="accuracy",
            )
            grid.fit(X, y)
            self.weighting_factor = grid.best_params_["metric_params"][
                "weighting_factor"
            ]

    def _preprocess(self, X):
        # private method for performing the transformations on
        # the test/training data. It extracts the subsequences
        # and then performs the shape descriptor function on
        # each subsequence.
        _X = self.sw.transform(X)
        # Feed X into the appropriate shape descriptor function
        _X = self._generate_shape_descriptors(_X)

        return _X

    def _predict_proba(self, X) -> np.ndarray:
        """Perform predictions on the testing data X.

        This function returns the probabilities for each class.

        Parameters
        ----------
        X : 3D np.ndarray
            The data to make predictions for, shape = (n_cases, n_channels,
            n_timepoints).

        Returns
        -------
        1D np.ndarray
            Predicted probabilities using the ordering in classes_, shape = (
            n_cases, n_classes_).
        """
        # Transform the test data in the same way as the training data.
        X = self._preprocess(X)

        # Classify the test data
        return self.knn.predict_proba(X)

    def _predict(self, X) -> np.ndarray:
        """Find predictions for all cases in X.

        Parameters
        ----------
        X : 3D np.ndarray
            The data to make predictions for, shape = (n_cases, n_channels,
            n_timepoints).

        Returns
        -------
        1D np.ndarray
            The predicted class labels shape = (n_cases).
        """
        # Transform the test data in the same way as the training data.
        X = self._preprocess(X)

        # Classify the test data
        return self.knn.predict(X)

    def _generate_shape_descriptors(self, data):
        """Generate shape descriptors.

        This function is used to convert a list of
        subsequences into a list of shape descriptors
        to be used for classification.
        """
        # Get the appropriate transformer objects
        if self.shape_descriptor_function != "compound":
            self.transformer = [self._get_transformer(self.shape_descriptor_function)]
        else:
            self.transformer = []
            self.transformer.extend(
                self._get_transformer(x) for x in self._shape_descriptor_functions
            )
            if len(self.transformer) != 2:
                raise ValueError(
                    "When using 'compound', "
                    + "shape_descriptor_functions must be a "
                    + "string array of length 2."
                )

        # To hold the result of each transformer
        trans_data = []

        # Apply each transformer on the set of subsequences
        for t in self.transformer:
            if t is None:
                # Do no transformations
                trans_data.append(data)
            else:
                # Do the transformation and extract the resulting data frame.
                t.fit(data)
                newData = t.transform(data)
                trans_data.append(newData)

        result = trans_data[0]
        for i in range(1, len(trans_data)):
            result = np.concatenate((result, trans_data[i]), axis=1)
        return result

    def _get_transformer(self, tName):
        """Extract the appropriate transformer.

        Requires self._metric_params, so only call after fit or in fit after these
        lines of code
        if self.metric_params is None:
            self._metric_params = {}
        else:
            self._metric_params = self.metric_params

        Parameters
        ----------
        self   : the ShapeDTW object.
        tName  : the name of the required transformer.

        Returns
        -------
        output : Base Transformer object corresponding to the class
                 (or classes if its a compound transformer) of the
                 required transformer. The transformer is
                 configured with the parameters given in self.metric_params.

        throws : ValueError if a shape descriptor doesn't exist.
        """
        parameters = self._metric_params
        tName = tName.lower()
        if parameters is None:
            parameters = {}
        parameters = {k.lower(): v for k, v in parameters.items()}
        self._check_metric_params(parameters)

        if tName == "raw":
            return None
        elif tName == "paa":
            num_intervals = parameters.get("num_intervals_paa")
            return PAA() if num_intervals is None else PAA(num_intervals)
        elif tName == "dwt":
            num_levels = parameters.get("num_levels_dwt")
            return (
                DWTTransformer() if num_levels is None else DWTTransformer(num_levels)
            )
        elif tName == "slope":
            num_intervals = parameters.get("num_intervals_slope")
            if num_intervals is None:
                return SlopeTransformer()
            return SlopeTransformer(num_intervals)
        elif tName == "derivative":
            return FunctionTransformer(func=slope_derivative_3d)
        elif tName == "hog1d":
            return self._get_hog_transformer(parameters)
        else:
            raise ValueError("Invalid shape descriptor function.")

    def _get_hog_transformer(self, parameters):
        num_intervals = parameters.get("num_intervals_hog1d")
        num_bins = parameters.get("num_bins_hog1d")
        scaling_factor = parameters.get("scaling_factor_hog1d")
        # All 3 paramaters are None
        if num_intervals is None and num_bins is None and scaling_factor is None:
            return HOG1DTransformer()
        # 2 parameters are None
        if num_intervals is None and num_bins is None:
            return HOG1DTransformer(scaling_factor=scaling_factor)
        if num_intervals is None and scaling_factor is None:
            return HOG1DTransformer(n_bins=num_bins)
        if num_bins is None and scaling_factor is None:
            return HOG1DTransformer(n_intervals=num_intervals)

        # 1 parameter is None
        if num_intervals is None:
            return HOG1DTransformer(scaling_factor=scaling_factor, n_bins=num_bins)
        if scaling_factor is None:
            return HOG1DTransformer(n_intervals=num_intervals, n_bins=num_bins)
        if num_bins is None:
            return HOG1DTransformer(
                scaling_factor=scaling_factor, n_intervals=num_intervals
            )
        # All parameters are given
        return HOG1DTransformer(
            n_intervals=num_intervals,
            n_bins=num_bins,
            scaling_factor=scaling_factor,
        )

    def _check_metric_params(self, parameters):
        """Check for an invalid metric_params."""
        valid_metric_params = [
            "num_intervals_paa",
            "num_levels_dwt",
            "num_intervals_slope",
            "num_intervals_hog1d",
            "num_bins_hog1d",
            "scaling_factor_hog1d",
            "weighting_factor",
        ]

        names = list(parameters.keys())

        for x in names:
            if x not in valid_metric_params:
                raise ValueError(
                    x
                    + " is not a valid metric parameter."
                    + "Make sure the shape descriptor function"
                    + " name is at the end of the metric "
                    + "parameter name."
                )
