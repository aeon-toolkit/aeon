"""tsfresh interface class."""

__maintainer__ = []
__all__ = ["TSFresh", "TSFreshRelevant"]

import numpy as np
import pandas as pd

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs


def _from_3d_numpy_to_long(arr):
    # Converting the numpy array to a long format DataFrame
    n_cases, n_channels, n_timepoints = arr.shape

    # Creating a DataFrame from the numpy array with multi-level index
    df = pd.DataFrame(arr.reshape(n_cases * n_channels, n_timepoints))
    df["case_index"] = np.repeat(np.arange(n_cases), n_channels)
    df["dimension"] = np.tile(np.arange(n_channels), n_cases)
    df = df.melt(
        id_vars=["case_index", "dimension"], var_name="time_index", value_name="value"
    )

    # Adjusting the column order and renaming columns
    df = df[["case_index", "time_index", "dimension", "value"]]
    df = df.rename(columns={"case_index": "index", "dimension": "column"})
    df["column"] = "dim_" + df["column"].astype(str)
    return df


class _TSFresh(BaseCollectionTransformer):
    """Base adapter class for tsfresh transformations."""

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "fit_is_empty": True,
        "python_dependencies": "tsfresh",
    }

    def __init__(
        self,
        default_fc_parameters=None,
        kind_to_fc_parameters=None,
        chunksize=None,
        n_jobs=1,
        show_warnings=False,
        disable_progressbar=False,
        impute_function=None,
        profiling=None,
        profiling_filename=None,
        profiling_sorting=None,
        distributor=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profiling = profiling
        self.profiling_sorting = profiling_sorting
        self.profiling_filename = profiling_filename
        self.distributor = distributor

        super().__init__()

        # _get_extraction_params should be after the init because this imports tsfresh
        # and the init checks for python version and tsfresh being present
        self.default_fc_parameters_ = self._get_extraction_params()

    def _get_extraction_params(self):
        """Set default parameters from tsfresh."""
        # make n_jobs compatible with scikit-learn
        n_jobs = self.n_jobs
        self.n_jobs = check_n_jobs(self.n_jobs)

        # lazy imports to avoid hard dependency
        from tsfresh.defaults import (
            CHUNKSIZE,
            DISABLE_PROGRESSBAR,
            N_PROCESSES,
            PROFILING,
            PROFILING_FILENAME,
            PROFILING_SORTING,
            SHOW_WARNINGS,
        )
        from tsfresh.feature_extraction.settings import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
            from_columns,
        )
        from tsfresh.utilities.dataframe_functions import impute

        # Set defaults from tsfresh
        extraction_params = {
            "n_jobs": N_PROCESSES,
            "chunksize": CHUNKSIZE,
            "show_warnings": SHOW_WARNINGS,
            "disable_progressbar": DISABLE_PROGRESSBAR,
            "impute_function": impute,
            "profiling_sorting": PROFILING_SORTING,
            "profiling_filename": PROFILING_FILENAME,
            "profile": PROFILING,
        }

        # Replace defaults with user defined parameters
        for name in extraction_params.keys():
            if hasattr(self, name):
                value = getattr(self, name)
                if value is not None:
                    extraction_params[name] = value

        self.n_jobs = n_jobs

        # Convert convenience string arguments to tsfresh parameters classes
        fc_param_lookup = {
            "minimal": MinimalFCParameters(),
            "efficient": EfficientFCParameters(),
            "comprehensive": ComprehensiveFCParameters(),
        }
        if isinstance(self.default_fc_parameters, str):
            if self.default_fc_parameters not in fc_param_lookup:
                raise ValueError(
                    f"If `default_fc_parameters` is passed as a "
                    f"string, "
                    f"it must be one of"
                    f" {fc_param_lookup.keys()}, but found: "
                    f"{self.default_fc_parameters}"
                )
            else:
                fc_parameters = fc_param_lookup[self.default_fc_parameters]
        else:
            fc_parameters = self.default_fc_parameters
        extraction_params["default_fc_parameters"] = fc_parameters

        # creates mapping from kind names to fc_parameter objects
        if self.kind_to_fc_parameters is not None:
            self.kind_to_fc_parameters_ = from_columns(self.kind_to_fc_parameters)
        else:
            self.kind_to_fc_parameters_ = self.kind_to_fc_parameters
        extraction_params["kind_to_fc_parameters"] = self.kind_to_fc_parameters_

        return extraction_params


class TSFresh(_TSFresh):
    """Transformer for extracting time series features via `tsfresh.extract_features`.

    Direct interface to `tsfresh.extract_features` [1] as an `aeon` transformer.

    Parameters
    ----------
    default_fc_parameters : str, FCParameters object or None,
        default=None = tsfresh default = "comprehensive"
        Specifies pre-defined feature sets to be extracted
        If str, should be in ["minimal", "efficient", "comprehensive"]
        See [3] for more details.
    kind_to_fc_parameters : list or None, default=None
        List containing strings specifying selected features to be extracted.
        The naming convention from tsfresh applies, i.e. the strings
        should be structured as:
        {time_series_name}__{feature_name}__{param name 1}_
        {param value 1}__[..]__{param name k}_{param value k}.
        See [2] for more details and [4] for viable options.
        Either default_fc_parameters or kind_to_fc_parameters
        should be passed. If both are passed, only features specified
        in kind_to_fc_parameters are extracted. If neither
        is passed, it calculates the "comprehensive"
        feature set.
    n_jobs : int, default=1
        The number of processes to use for parallelization.
        If zero, no parallelization is used.
    chunksize : None or int, default=None
        The size of one chunk that is submitted to the worker
        process for the parallelisation.  Where one chunk is defined as a
        singular time series for one id and one kind. If you set the chunksize
        to 10, then it means that one task is to calculate all features for 10
        time series.  If it is set it to None, depending on distributor,
        heuristics are used to find the optimal chunksize. If you get out of
        memory exceptions, you can try it with the dask distributor and a
        smaller chunksize.
    show_warnings : bool, default=True
        Show warnings during the feature extraction (needed for debugging of
        calculators).
    disable_progressbar : bool, default=False
        Do not show a progressbar while doing the calculation.
    impute_function : None or Callable, default=None
        None, if no imputing should happen or the function to call for
        imputing the result dataframe. Imputing will never happen on the input data.
    profiling : bool, default=None
        Turn on profiling during feature extraction.
    profiling_sorting : basestring, default=None
        How to sort the profiling results (see the documentation of the tsfresh
        profiling package for more information).
    profiling_filename : basestring, default=None
        Where to save the profiling results.
    distributor : distributor class, default=None
        Advanced parameter: set this to a class name that you want to use as a
        distributor. See the tsfresh package utilities/distribution.py for more
        information.
        Leave to None, if you want TSFresh to choose the best distributor.

    References
    ----------
    .. [1]  https://github.com/blue-yonder/tsfresh
    .. [2]  https://tsfresh.readthedocs.io/en/v0.1.2/text/feature_naming.html
    .. [3]  https://tsfresh.readthedocs.io/en/latest/text/
            feature_extraction_settings.html
    .. [4]  https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
            #module-tsfresh.feature_extraction.feature_calculators
    .. [5] Christ, M., Braun, N., Neuffer, J., and Kempa-Liehr A.W. (2018). Time Series
        FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh -- A Python
        package). Neurocomputing 307 (2018) 72-77

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from aeon.datasets import load_arrow_head
    >>> from aeon.transformations.collection.feature_based import (
    ...     TSFresh
    ... )
    >>> X, y = load_arrow_head()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> ts_eff = TSFresh(
    ...     default_fc_parameters="efficient", disable_progressbar=True
    ... ) # doctest: +SKIP
    >>> X_transform1 = ts_eff.fit_transform(X_train) # doctest: +SKIP
    >>> features_to_calc = [
    ...     "dim_0__quantile__q_0.6",
    ...     "dim_0__longest_strike_above_mean",
    ...     "dim_0__variance",
    ... ]
    >>> ts_custom = TSFresh(
    ...     kind_to_fc_parameters=features_to_calc, disable_progressbar=True
    ... ) # doctest: +SKIP
    >>> X_transform2 = ts_custom.fit_transform(X_train) # doctest: +SKIP
    """

    def __init__(
        self,
        default_fc_parameters="efficient",
        kind_to_fc_parameters=None,
        chunksize=None,
        n_jobs=1,
        show_warnings=False,
        disable_progressbar=True,
        impute_function=None,
        profiling=None,
        profiling_filename=None,
        profiling_sorting=None,
        distributor=None,
    ):
        super().__init__(
            default_fc_parameters=default_fc_parameters,
            kind_to_fc_parameters=kind_to_fc_parameters,
            chunksize=chunksize,
            n_jobs=n_jobs,
            show_warnings=show_warnings,
            disable_progressbar=disable_progressbar,
            impute_function=impute_function,
            profiling=profiling,
            profiling_filename=profiling_filename,
            profiling_sorting=profiling_sorting,
            distributor=distributor,
        )
        self._get_names()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : 3D numpy array of shape (n_cases, n_channels, n_features)
            input time series collection.
        y : ignored argument for interface compatibility

        Returns
        -------
        X : 3D numpy array of shape (n_cases, n_channels, n_features)
            input time series collection.
            transformed version of X
        """
        Xt = _from_3d_numpy_to_long(X)

        # lazy imports to avoid hard dependency
        from tsfresh import extract_features

        Xt = extract_features(
            Xt,
            column_id="index",
            column_value="value",
            column_kind="column",
            column_sort="time_index",
            **self.default_fc_parameters_,
        )
        return Xt.to_numpy()

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        features_to_calc = [
            "dim_0__quantile__q_0.6",
            "dim_0__longest_strike_above_mean",
            "dim_0__variance",
        ]

        return [
            {
                "disable_progressbar": True,
                "show_warnings": False,
                "default_fc_parameters": "minimal",
            },
            {
                "disable_progressbar": True,
                "show_warnings": False,
                "kind_to_fc_parameters": features_to_calc,
            },
        ]

    def _get_names(self):
        """Hack to get the feature names prior to transform."""
        from tsfresh import extract_features

        X = np.random.random((2, 1, 30))
        Xt = _from_3d_numpy_to_long(X)
        Xt = extract_features(
            Xt,
            column_id="index",
            column_value="value",
            column_kind="column",
            column_sort="time_index",
            **self.default_fc_parameters_,
        )
        # Get the list of feature names
        self.names = Xt.columns.tolist()


class TSFreshRelevant(_TSFresh):
    """Transformer for extracting time series features via `tsfresh.extract_features`.

    Direct interface to `tsfresh.extract_features` [1] followed by the tsfresh
    FeatureSelector class as an `aeon` transformer.

    Parameters
    ----------
    default_fc_parameters : str, FCParameters object or None,
        default=None = tsfresh default = "comprehensive"
        Specifies pre-defined feature sets to be extracted
        If str, should be in ["minimal", "efficient", "comprehensive"]
        See [3] for more details.
    kind_to_fc_parameters : list or None, default=None
        List containing strings specifying selected features to be extracted.
        The naming convention from tsfresh applies, i.e. the strings
        should be structured as:
        {time_series_name}__{feature_name}__{param name 1}_
        {param value 1}__[..]__{param name k}_{param value k}.
        See [2] for more details and [4] for viable options.
        Either default_fc_parameters or kind_to_fc_parameters
        should be passed. If both are passed, only features specified
        in kind_to_fc_parameters are extracted. If neither
        is passed, it calculates the "comprehensive"
        feature set.
    n_jobs : int, default=1
        The number of processes to use for parallelization.
        If zero, no parallelization is used.
    chunksize : None or int, default=None
        The size of one chunk that is submitted to the worker
        process for the parallelisation.  Where one chunk is defined as a
        singular time series for one id and one kind. If you set the chunksize
        to 10, then it means that one task is to calculate all features for 10
        time series.  If it is set it to None, depending on distributor,
        heuristics are used to find the optimal chunksize. If you get out of
        memory exceptions, you can try it with the dask distributor and a
        smaller chunksize.
    show_warnings : bool, default=False
        Show warnings during the feature extraction (needed for debugging of
        calculators).
    disable_progressbar : bool, default=False
        Do not show a progressbar while doing the calculation.
    impute_function : None or Callable, default=None
        None, if no imputing should happen or the function to call for
        imputing the result dataframe. Imputing will never happen on the input data.
    profiling : bool, default=None
        Turn on profiling during feature extraction.
    profiling_sorting : basestring, default=None
        How to sort the profiling results (see the documentation of the tsfresh
        profiling package for more information).
    profiling_filename : basestring, default=None
        Where to save the profiling results.
    distributor : distributor class, default=None
        Advanced parameter: set this to a class name that you want to use as a
        distributor. See the tsfresh package utilities/distribution.py for more
        information.
        Leave to None, if you want TSFresh to choose the best distributor.
    test_for_binary_target_binary_feature : str or None, default=None
        Which test to be used for binary target, binary feature (currently unused).
    test_for_binary_target_real_feature : str or None, default=None
        Which test to be used for binary target, real feature.
    test_for_real_target_binary_feature : str or None, default=None
        Which test to be used for real target, binary feature (currently unused).
    test_for_real_target_binary_feature : str or None, default=None
        Which test to be used for real target, real feature (currently unused)
    fdr_level: float or None, default=None
        The FDR level that should be respected, this is the theoretical expected
        percentage of irrelevant features among all created features.
    hypotheses_independent: bool or None, default=None
        Can the significance of the features be assumed to be independent? Normally,
        this should be set to False as the features are never independent (e.g. mean
        and median)
    ml_task: sre, default="auto"
        The intended machine learning task. Either `'classification'`, `'regression'`
        or `'auto'`.
        Defaults to `'auto'`, meaning the intended task is inferred from `y`.
        If `y` has a boolean, integer or object dtype, the task is assumed to be
        classification, else regression.

    References
    ----------
    .. [1]  https://github.com/blue-yonder/tsfresh
    .. [2]  https://tsfresh.readthedocs.io/en/v0.1.2/text/feature_naming.html
    .. [3]  https://tsfresh.readthedocs.io/en/latest/text/
            feature_extraction_settings.html
    .. [4]  https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
            #module-tsfresh.feature_extraction.feature_calculators
    .. [5] Christ, M., Braun, N., Neuffer, J., and Kempa-Liehr A.W. (2018). Time Series
        FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh -- A Python
        package). Neurocomputing 307 (2018) 72-77

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from aeon.datasets import load_arrow_head
    >>> from aeon.transformations.collection.feature_based import (
    ...     TSFreshRelevant
    ... )
    >>> X, y = load_arrow_head()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> ts_eff = TSFreshRelevant(
    ...     default_fc_parameters="efficient", disable_progressbar=True
    ... ) # doctest: +SKIP
    >>> X_transform1 = ts_eff.fit_transform(X_train, y_train) # doctest: +SKIP
    >>> features_to_calc = [
    ...     "dim_0__quantile__q_0.6",
    ...     "dim_0__longest_strike_above_mean",
    ...     "dim_0__variance",
    ... ]
    >>> ts_custom = TSFreshRelevant(
    ...     kind_to_fc_parameters=features_to_calc, disable_progressbar=True
    ... ) # doctest: +SKIP
    >>> X_transform2 = ts_custom.fit_transform(X_train, y_train) # doctest: +SKIP
    """

    _tags = {
        "requires_y": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": False,
    }

    def __init__(
        self,
        default_fc_parameters="efficient",
        kind_to_fc_parameters=None,
        chunksize=None,
        n_jobs=1,
        show_warnings=False,
        disable_progressbar=True,
        impute_function=None,
        profiling=None,
        profiling_filename=None,
        profiling_sorting=None,
        distributor=None,
        test_for_binary_target_binary_feature=None,
        test_for_binary_target_real_feature=None,
        test_for_real_target_binary_feature=None,
        test_for_real_target_real_feature=None,
        fdr_level=None,
        hypotheses_independent=None,
        ml_task="auto",
    ):
        super().__init__(
            default_fc_parameters=default_fc_parameters,
            kind_to_fc_parameters=kind_to_fc_parameters,
            chunksize=chunksize,
            n_jobs=n_jobs,
            show_warnings=show_warnings,
            disable_progressbar=disable_progressbar,
            impute_function=impute_function,
            profiling=profiling,
            profiling_filename=profiling_filename,
            profiling_sorting=profiling_sorting,
            distributor=distributor,
        )

        self.test_for_binary_target_binary_feature = (
            test_for_binary_target_binary_feature
        )
        self.test_for_binary_target_real_feature = test_for_binary_target_real_feature
        self.test_for_real_target_binary_feature = test_for_real_target_binary_feature
        self.test_for_real_target_real_feature = test_for_real_target_real_feature
        self.fdr_level = fdr_level
        self.hypotheses_independent = hypotheses_independent
        self.ml_task = ml_task

        self.default_fs_parameters_ = self._get_selection_params()
        self.names_ = []

    def _get_selection_params(self):
        """Set default values from tsfresh."""
        # lazy imports to avoid hard dependency
        from tsfresh.defaults import (
            FDR_LEVEL,
            HYPOTHESES_INDEPENDENT,
            TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
            TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            TEST_FOR_REAL_TARGET_BINARY_FEATURE,
            TEST_FOR_REAL_TARGET_REAL_FEATURE,
        )

        # Set defaults
        selection_params = {
            "test_for_binary_target_binary_feature": TEST_FOR_BINARY_TARGET_BINARY_FEATURE,  # noqa: E501
            "test_for_binary_target_real_feature": TEST_FOR_BINARY_TARGET_REAL_FEATURE,
            "test_for_real_target_binary_feature": TEST_FOR_REAL_TARGET_BINARY_FEATURE,
            "test_for_real_target_real_feature": TEST_FOR_REAL_TARGET_REAL_FEATURE,
            "fdr_level": FDR_LEVEL,
            "hypotheses_independent": HYPOTHESES_INDEPENDENT,
        }

        # Replace defaults with user defined parameters
        for name in selection_params.keys():
            value = getattr(self, name)
            if value is not None:
                selection_params[name] = value

        return selection_params

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            collection of time series to transform
        y : Series, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # lazy imports to avoid hard dependency
        from tsfresh.transformers.feature_selector import FeatureSelector

        self.extractor_ = TSFresh(
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            chunksize=self.chunksize,
            n_jobs=self.n_jobs,
            show_warnings=self.show_warnings,
            disable_progressbar=self.disable_progressbar,
            profiling=self.profiling,
            profiling_filename=self.profiling_filename,
            profiling_sorting=self.profiling_sorting,
        )

        self.selector_ = FeatureSelector(
            n_jobs=self.default_fc_parameters_["n_jobs"],
            chunksize=self.default_fc_parameters_["chunksize"],
            ml_task=self.ml_task,
            **self.default_fs_parameters_,
        )

        Xt = self.extractor_.fit_transform(X)
        Xt = self.selector_.fit_transform(Xt, y)
        self.names_ = self.selector_.relevant_features
        return Xt

    def _fit(self, X, y=None):
        """Fit.

        Parameters
        ----------
        X : pd.DataFrame
            nested pandas DataFrame of shape [n_samples, n_columns]
        y : pd.Series or np.array
            Target variable

        Returns
        -------
        self : an instance of self
        """
        # lazy imports to avoid hard dependency
        from tsfresh.transformers.feature_selector import FeatureSelector

        self.extractor_ = TSFresh(
            default_fc_parameters=self.default_fc_parameters,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            chunksize=self.chunksize,
            n_jobs=self.n_jobs,
            show_warnings=self.show_warnings,
            disable_progressbar=self.disable_progressbar,
            profiling=self.profiling,
            profiling_filename=self.profiling_filename,
            profiling_sorting=self.profiling_sorting,
        )

        self.selector_ = FeatureSelector(
            n_jobs=self.default_fc_parameters_["n_jobs"],
            chunksize=self.default_fc_parameters_["chunksize"],
            ml_task=self.ml_task,
            **self.default_fs_parameters_,
        )

        Xt = self.extractor_.fit_transform(X)
        self.selector_.fit(Xt, y)
        self.names_ = self.selector_.relevant_features
        return self

    def _transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : np.ndarray
        y : None
            Ignored

        Returns
        -------
        Xt : np.ndarray
          Transformed data
        """
        Xt = self.extractor_.transform(X)
        Xt = self.selector_.transform(Xt)
        return Xt

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {
            "default_fc_parameters": "efficient",
            "disable_progressbar": True,
            "show_warnings": False,
            "fdr_level": 0.01,
        }
        return params
