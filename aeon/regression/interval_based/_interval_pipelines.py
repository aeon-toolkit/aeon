# -*- coding: utf-8 -*-
class RandomIntervalRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Random Interval Classifier.

    This classifier simply transforms the input data using the RandomIntervals
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    n_intervals : int, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.
    interval_transformers : transformer or list of transformers, default=None,
        Transformer(s) used to extract features from each interval. If None, defaults to
        the Catch22 transformer.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Rotation Forest with 200 trees.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes)
        Holds the label for each class.

    See Also
    --------
    RandomIntervals
    """

    def __init__(
        self,
        n_intervals=100,
        interval_transformers=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_intervals = n_intervals
        self.interval_transformers = interval_transformers
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        super(RandomIntervalRegressor, self).__init__()

    def fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=3
        )
        X = self._convert_X(X)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        interval_transformers = (
            Catch22Transformer(catch24=True, outlier_norm=True, replace_nans=True)
            if self.interval_transformers is None
            else self.interval_transformers
        )

        self._transformer = RandomIntervalTransformer(
            n_intervals=self.n_intervals,
            features=interval_transformers,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )

        self._estimator = _clone_estimator(
            RandomForestRegressor() if self.estimator is None else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        return self._estimator.predict(self._transformer.transform(X))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {
                "n_intervals": 5,
                "estimator": RandomForestRegressor(n_estimators=10),
                "interval_transformers": Catch22Transformer(
                    catch24=True,
                    replace_nans=True,
                    features=(
                        "Mean",
                        "DN_HistogramMode_5",
                        "SB_BinaryStats_mean_longstretch1",
                    ),
                ),
            }
        else:
            return {
                "n_intervals": 3,
                "estimator": RandomForestRegressor(n_estimators=2),
                "interval_transformers": Catch22Transformer(
                    catch24=True,
                    replace_nans=True,
                    features=(
                        "Mean",
                        "DN_HistogramMode_5",
                        "SB_BinaryStats_mean_longstretch1",
                    ),
                ),
            }
