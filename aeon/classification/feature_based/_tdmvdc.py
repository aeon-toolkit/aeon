"""Tracking Differentiator-based Multiview Dilated Characteristics (TDMVDC).

Enhances the discriminative power of time series data by applying a
combination of tracking differentiator-based transformations and multiview
dilated feature extraction. Leverages advanced feature engineering techniques
to capture both the original and differential characteristics of time series at
multiple temporal resolutions. Integrates feature selection and ensemble
learning to improve classification accuracy and robustness across diverse time
series datasets.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.feature_based import (
    TSFresh,
    hard_voting,
    series_set_dilation,
    series_transform,
)
from aeon.utils.validation import check_n_jobs

__all__ = ["TDMVDCClassifier"]


class TDMVDCClassifier(BaseClassifier):
    """Tracking Differentiator-based Multiview Dilated Characteristics classifier.

    The TDMVDCClassifier is an advanced ensemble classifier tailored for
    time series classification tasks. It operates by transforming the
    input time series data through a tracking differentiator, generating
    three distinct views: the original signal, the first-order differential,
    and the second-order differential. Each of these views is further processed
    using a set of dilation rates, allowing the model to capture temporal
    dependencies and patterns at multiple scales.

    For each dilated view, the classifier extracts a comprehensive set of
    features using the TSFresh feature extraction framework. These features are
    then evaluated using ANOVA F-values to determine their relevance to the
    classification task. Multiple classifiers are constructed, each trained on a
    different proportion of the most informative features, as determined by the
    feature selection process. The ensemble of classifiers provides robust
    predictions by aggregating their outputs through a hard voting mechanism,
    which enhances generalization and reduces the risk of overfitting.

    The TDMVDCClassifier is highly parallelized, supporting multi-threaded feature
    extraction and model training to efficiently handle large datasets. Its design
    is particularly effective for time series problems where both the original and
    differential characteristics of the data, as well as multiscale temporal patterns,
    are important for accurate classification.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Specifies the set of TSFresh features to extract. Options include "minimal",
        "efficient", or "comprehensive".
    k1 : float, default=2
        Filter parameter for the first tracking differentiator, controlling the
        generation of first-order differential series.
    k2 : float, default=2
        Filter parameter for the second tracking differentiator, controlling the
        generation of second-order differential series.
    feature_store_ratios : list, default=None
        List of feature retention ratios for different feature selectors.
        If None, defaults to [0.1, 0.2, 0.3, 0.4, 0.5].
    n_jobs : int, default=1
        Number of parallel jobs to run for feature extraction and model training.
        "-1" uses all available processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specifies the parallelization backend for joblib. Options include "loky",
        "multiprocessing", "threading", or a custom backend.

    Attributes
    ----------
    n_classes_ : int
        Number of unique classes in the training data.
    classes_ : ndarray of shape (n_classes_)
        Array of class labels.
    clfList_ : list
        List of trained classifier pipelines for each feature subset.
    dList_ : ndarray
        Array of dilation rates used for multiscale feature extraction.
    tsFreshListR_ : list
        List of TSFresh feature extractors for original signal at each dilation rate.
    tsFreshListF_ : list
        List of TSFresh feature extractors for the first-order differential signal
        at each dilation rate.
    tsFreshListS_ : list
        List of TSFresh feature extractors for the second-order differential signal
        at each dilation rate.
    scoreRFS_ : ndarray
        Array of ANOVA F-values for all extracted features, used for feature selection.

    Notes
    -----
    The TDMVDCClassifier is particularly effective for time series datasets where
    capturing both the original and differential dynamics, as well as multiscale
    temporal features, is crucial for distinguishing between classes. Its ensemble
    approach and feature selection strategy help to mitigate overfitting and improve
    predictive performance on complex datasets.

    For the algorithm details, see [1]_.

    References
    ----------
    .. [1] Changchun He, and Xin Huo. "Tracking Differentiator-based Multiview Dilated
        Characteristics for Time Series Classification." in The 22nd IEEE International
        Conference on Industrial Informatics (INDIN2024) (2024).
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "feature",
        "python_dependencies": "tsfresh",
    }

    def __init__(
        self,
        default_fc_parameters="efficient",
        k1=2,
        k2=2,
        feature_store_ratios=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.k1 = k1
        self.k2 = k2
        self.feature_store_ratios = (
            feature_store_ratios
            if feature_store_ratios is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5]
        )
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        super().__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X, y).

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        # Initialization of dilation rate parameters
        n_timepoints = X.shape[2]
        d_min = 0
        d_max = int(np.log2(n_timepoints - 1) - 3)
        d_max = np.min([5, d_max])
        self.dList_ = 2 ** np.arange(d_min, d_max + 1)

        # Differential transformations by tracking differentiator
        X_F = series_transform(X, mode=1, k1=self.k1)
        X_S = series_transform(X, mode=2, k1=self.k1, k2=self.k2)

        # Feature extraction
        self.tsFreshListR_ = []
        self.tsFreshListF_ = []
        self.tsFreshListS_ = []

        # Train feature sets
        RXList = []
        FXList = []
        SXList = []

        # Use parallel processing for feature extraction
        threads_to_use = check_n_jobs(self.n_jobs)

        # Extract features for each dilation rate in parallel
        results = Parallel(
            n_jobs=threads_to_use, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._extract_features_for_dilation)(X, X_F, X_S, d_rate, y)
            for d_rate in self.dList_
        )

        # Unpack results
        for tsFreshR, tsFreshF, tsFreshS, RX, FX, SX in results:
            self.tsFreshListR_.append(tsFreshR)
            self.tsFreshListF_.append(tsFreshF)
            self.tsFreshListS_.append(tsFreshS)
            RXList.append(RX)
            FXList.append(FX)
            SXList.append(SX)

        # Concatenating all the dilated features
        RX = np.hstack(RXList)
        FX = np.hstack(FXList)
        SX = np.hstack(SXList)

        # Computing feature scores
        self.scoreRFS_ = f_classif(np.hstack((RX, FX, SX)), y)[0]
        self.scoreRFS_[np.isnan(self.scoreRFS_)] = 0

        # Training the classifier on each view
        self.clfList_ = []

        # Train classifiers for each feature ratio in parallel
        self.clfList_ = Parallel(
            n_jobs=threads_to_use, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._train_classifier_for_ratio)(RX, FX, SX, y, ratio)
            for ratio in self.feature_store_ratios
        )

        return self

    def _extract_features_for_dilation(self, X, X_F, X_S, d_rate, y):
        """Extract features for a specific dilation rate.

        Parameters
        ----------
        X : 3D np.ndarray
            Original signal.
        X_F : 3D np.ndarray
            First-order differential signal.
        X_S : 3D np.ndarray
            Second-order differential signal.
        d_rate : int
            Dilation rate.
        y : array-like
            Class labels.

        Returns
        -------
        tuple
            Tuple containing feature extractors and extracted features.
        """
        # Dilation Mapping
        RX_E = series_set_dilation(X, d_rate)
        FX_E = series_set_dilation(X_F, d_rate)
        SX_E = series_set_dilation(X_S, d_rate)

        # Extracting the TSFresh features
        tsFreshR = TSFresh(
            default_fc_parameters=self.default_fc_parameters, n_jobs=self._n_jobs
        )
        tsFreshR.fit(RX_E, y)
        RX = np.array(tsFreshR.transform(RX_E))

        tsFreshF = TSFresh(
            default_fc_parameters=self.default_fc_parameters, n_jobs=self._n_jobs
        )
        tsFreshF.fit(FX_E, y)
        FX = np.array(tsFreshF.transform(FX_E))

        tsFreshS = TSFresh(
            default_fc_parameters=self.default_fc_parameters, n_jobs=self._n_jobs
        )
        tsFreshS.fit(SX_E, y)
        SX = np.array(tsFreshS.transform(SX_E))

        return tsFreshR, tsFreshF, tsFreshS, RX, FX, SX

    def _train_classifier_for_ratio(self, RX, FX, SX, y, ratio):
        """Train a classifier for a specific feature ratio.

        Parameters
        ----------
        RX : 2D np.ndarray
            Features from original signal.
        FX : 2D np.ndarray
            Features from first-order differential signal.
        SX : 2D np.ndarray
            Features from second-order differential signal.
        y : array-like
            Class labels.
        ratio : float
            Feature store ratio.

        Returns
        -------
        Pipeline
            Trained classifier pipeline.
        """
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
            ]
        )

        bestIndex_ = np.argsort(self.scoreRFS_)[::-1][
            0 : int(len(self.scoreRFS_) * ratio)
        ]

        clf.fit(np.hstack((RX, FX, SX))[:, bestIndex_], y)

        return clf

    def _predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        X_F = series_transform(X, mode=1, k1=self.k1)
        X_S = series_transform(X, mode=2, k1=self.k1, k2=self.k2)

        # Use parallel processing for feature extraction
        threads_to_use = check_n_jobs(self.n_jobs)

        # Extract features for each dilation rate in parallel
        results = Parallel(
            n_jobs=threads_to_use, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._extract_test_features_for_dilation)(X, X_F, X_S, d_rate, i)
            for i, d_rate in enumerate(self.dList_)
        )

        # Unpack results
        RXList = []
        FXList = []
        SXList = []

        for RX, FX, SX in results:
            RXList.append(RX)
            FXList.append(FX)
            SXList.append(SX)

        # Concatenating all the dilated features
        RX = np.hstack(RXList)
        FX = np.hstack(FXList)
        SX = np.hstack(SXList)

        # Predict in parallel for each classifier
        PYList = Parallel(
            n_jobs=threads_to_use, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._predict_with_classifier)(RX, FX, SX, i, ratio)
            for i, ratio in enumerate(self.feature_store_ratios)
        )

        # Convert to numpy array for voting
        PYList = np.vstack(PYList)

        # Final prediction by hard voting
        PYV = hard_voting(PYList)

        return PYV

    def _extract_test_features_for_dilation(self, X, X_F, X_S, d_rate, i):
        """Extract test features for a specific dilation rate.

        Parameters
        ----------
        X : 3D np.ndarray
            Original signal.
        X_F : 3D np.ndarray
            First-order differential signal.
        X_S : 3D np.ndarray
            Second-order differential signal.
        d_rate : int
            Dilation rate.
        i : int
            Index of dilation rate.

        Returns
        -------
        tuple
            Tuple containing extracted features.
        """
        # Dilation Mapping
        RX_E = series_set_dilation(X, d_rate)
        FX_E = series_set_dilation(X_F, d_rate)
        SX_E = series_set_dilation(X_S, d_rate)

        # Extracting the TSFresh features
        tsFreshR = self.tsFreshListR_[i]
        RX = np.array(tsFreshR.transform(RX_E))

        tsFreshF = self.tsFreshListF_[i]
        FX = np.array(tsFreshF.transform(FX_E))

        tsFreshS = self.tsFreshListS_[i]
        SX = np.array(tsFreshS.transform(SX_E))

        return RX, FX, SX

    def _predict_with_classifier(self, RX, FX, SX, i, ratio):
        """Make predictions using a specific classifier.

        Parameters
        ----------
        RX : 2D np.ndarray
            Features from original signal.
        FX : 2D np.ndarray
            Features from first-order differential signal.
        SX : 2D np.ndarray
            Features from second-order differential signal.
        i : int
            Index of classifier.
        ratio : float
            Feature store ratio.

        Returns
        -------
        ndarray
            Predicted labels.
        """
        clf = self.clfList_[i]
        bestIndex_ = np.argsort(self.scoreRFS_)[::-1][
            0 : int(len(self.scoreRFS_) * ratio)
        ]

        return clf.predict(np.hstack((RX, FX, SX))[:, bestIndex_])

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
        """
        return {
            "k1": 2,
            "k2": 2,
            "feature_store_ratios": [0.1, 0.2, 0.3, 0.4, 0.5],
            "default_fc_parameters": "minimal",
        }
