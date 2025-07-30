"""Tracking Differentiator-based Multiview Dilated Characteristics (TDMVDC).

Ensemble classifier using TSFresh features and ANOVA,
with RidgeClassifierCV and hard voting.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.feature_based import TSFresh
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
        self.feature_store_ratios = feature_store_ratios
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        super().__init__()

    def _series_set_dilation(self, seriesX, d_rate=1):
        """
        Map each series of the time series set by dilation mapping.

        Should have the same dilation rate.

        Parameters
        ----------
        seriesX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The set of three dimensional time series set to be dilated.
        d_rate : int, default=1
            Dilation rate.

        References
        ----------
        .. [1] P. Schaefer and U. Leser, "WEASEL 2.0: a random dilated dictionary
        transform for fast, accurate and memory constrained time series classification"
        Machine Learning, vol. 112, no. 12, pp. 4763–4788, Dec.(2024).
        """
        n_cases, n_channels, _ = seriesX.shape[:]
        seriesXE = np.zeros_like(seriesX)  # Initializing the dilated time series set
        for i in range(n_cases):
            for j in range(n_channels):
                series_ = []
                for d in range(d_rate):
                    series_.append(seriesX[i, j, d::d_rate])
                seriesXE[i, j, :] = np.hstack(series_)
        return seriesXE  # Return the dilated time series set

    def _fhan(self, x1, x2, r, h0):
        """
        Calculate differential signal based on optimal control.

        Parameters
        ----------
        x1 : float
            State 1 of the observer.
        x2 : float
            State 2 of the observer.
        r: float
            Velocity factor used to control tracking speed.
        h0 : float
            Step size.

        References
        ----------
        .. [1] J. Han, "From PID to active disturbance rejection control" IEEE Trans.
        Ind. Electron., vol. 56, no. 3, pp. 900-906, Mar. (2009)..
        """
        d = r * h0
        d0 = d * h0
        y = x1 + h0 * x2  # Computing the differential signal
        a0 = np.sqrt(d * d + 8 * r * np.abs(y))
        if np.abs(y) > d0:
            a = x2 + (a0 - d) / 2.0 * np.sign(y)
        else:
            a = x2 + y / h0
        if np.abs(a) <= d:  # Computing the input u of observer
            u = -r * a / d
        else:
            u = -r * np.sign(a)
        return u, y  # Return input u of observer, and differential signal y

    def _td(self, signal, r=100, k=3, h=1):
        """
        Compute a differential signal using the tracking differentiator.

        with an adjustable filter factor.

        Parameters
        ----------
        signal : 1D np.ndarray of shape = [n_timepoints]
            Original time series
        r : float
            Velocity factor used to control tracking speed.
        k: float
            Filter factor.
        h : float
            Step size.

        References
        ----------
        .. [1] J. Han, "From PID to active disturbance rejection control" IEEE Trans.
        Ind. Electron., vol. 56, no. 3, pp. 900-906, Mar. (2009)..
        """
        x1 = signal[0]  # Initializing state 1
        x2 = -(signal[1] - signal[0]) / h  # Initializing state 2
        h0 = k * h
        signalTD = np.zeros(len(signal))
        dSignal = np.zeros(len(signal))
        for i in range(len(signal)):
            v = signal[i]
            x1k = x1
            x2k = x2
            x1 = x1k + h * x2k  # Update state 1
            u, y = self._fhan(
                x1k - v, x2k, r, h0
            )  # Update input u of observer and differential signal y
            x2 = x2k + h * u  # Update state 2
            dSignal[i] = y
            signalTD[i] = x1
        dSignal = -dSignal / h0  # Scale transform
        return dSignal[1:]  # Return the differential signal

    def _series_transform(self, seriesX, mode=1, k1=2, k2=2):
        """
        Transform each series of the time series set using a tracking differentiator.

        with an adjustable filter factor.

        Parameters
        ----------
        seriesX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The set of three dimensional time series set to be dilated.
        mode : int, default=1
            The flag bit of a first-order or second-order derivative is used.
            Computing the first-order derivative when mode=1,
            and computing the second-order derivative when mode=2
        k1 : float, default=2
            Filter factor 1 of the tracking differentiator 1.
        k2 : float, default=2
            Filter factor 2 of the tracking differentiator 2.
            This parameter is invalid when mode=2.

        References
        ----------
        .. [1] J. Han, "From PID to active disturbance rejection control" IEEE Trans.
        Ind. Electron., vol. 56, no. 3, pp. 900-906, Mar. (2009)..
        """
        from sklearn.preprocessing import scale

        n_cases, n_channels, n_timepoints = seriesX.shape[:]
        if mode == 1:  # First-order derivative
            seriesFX = np.zeros((n_cases, n_channels, n_timepoints - 1))
            for i in range(n_cases):
                for j in range(n_channels):
                    seriesFX[i, j, :] = self._td(seriesX[i, j, :], k=k1)
                    seriesFX[i, j, :] = scale(seriesFX[i, j, :])
            return seriesFX  # Return the first-order differential time series set
        if mode == 2:  # Second-order derivative
            seriesSX = np.zeros((n_cases, n_channels, n_timepoints - 2))
            for i in range(n_cases):
                for j in range(n_channels):
                    seriesF_ = self._td(seriesX[i, j, :], k=k1)
                    seriesSX[i, j, :] = self._td(seriesF_, k=k2)
                    seriesSX[i, j, :] = scale(seriesSX[i, j, :])
            return seriesSX  # Return the second-order differential time series set

    def _hard_voting(self, testYList):
        """
        Obtain the predicted labels by hard voting.

        to process the labels matrix from multiple classifiers.

        Parameters
        ----------
        testYList : 2D np.ndarray of shape = [n_classifierss, n_cases]
        """
        uniqueY = np.unique(testYList)  # Holds the label for each class
        n_classes = len(uniqueY)  # Number of classes
        n_classifiers, n_cases = testYList.shape[
            :
        ]  # Number of classifiers, Number of cases
        testVY = np.zeros(
            n_cases, int
        )  # 1 * n_cases, Initializing the predicted labels
        testWeightArray = np.zeros(
            (n_classes, n_cases)
        )  # n_classes * n_cases, Label weight matrix for samples
        for i in range(n_cases):
            for j in range(n_classifiers):
                label_ = testYList[j, i]
                index_ = np.arange(n_classes)[uniqueY == label_]
                testWeightArray[
                    index_, i
                ] += 1  # The label weight for the sample is + 1
        for i in range(n_cases):  # Predicting each sample label
            testVY[i] = uniqueY[
                np.argmax(testWeightArray[:, i])
            ]  # The label is predicted to be the most weighted
        return testVY  # return the predicted labels

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
        X_F = self._series_transform(X, mode=1, k1=self.k1)
        X_S = self._series_transform(X, mode=2, k1=self.k1, k2=self.k2)

        # Feature extraction
        self.tsFreshListR_ = []
        self.tsFreshListF_ = []
        self.tsFreshListS_ = []

        # Train feature sets
        RXList = []
        FXList = []
        SXList = []

        # Use parallel processing for feature extraction
        self._n_jobs = check_n_jobs(self.n_jobs)

        # Extract features for each dilation rate in parallel
        results = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
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
        # Ensure feature_store_ratios is set
        feature_store_ratios = (
            self.feature_store_ratios
            if self.feature_store_ratios is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5]
        )

        self.clfList_ = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._train_classifier_for_ratio)(RX, FX, SX, y, ratio)
            for ratio in feature_store_ratios
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
        RX_E = self._series_set_dilation(X, d_rate)
        FX_E = self._series_set_dilation(X_F, d_rate)
        SX_E = self._series_set_dilation(X_S, d_rate)

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
        X_F = self._series_transform(X, mode=1, k1=self.k1)
        X_S = self._series_transform(X, mode=2, k1=self.k1, k2=self.k2)

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

        # Ensure feature_store_ratios is set
        feature_store_ratios = (
            self.feature_store_ratios
            if self.feature_store_ratios is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5]
        )

        # Predict in parallel for each classifier
        PYList = Parallel(
            n_jobs=threads_to_use, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._predict_with_classifier)(RX, FX, SX, i, ratio)
            for i, ratio in enumerate(feature_store_ratios)
        )

        # Convert to numpy array for voting
        PYList = np.vstack(PYList)

        # Final prediction by hard voting
        PYV = self._hard_voting(PYList)

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
        RX_E = self._series_set_dilation(X, d_rate)
        FX_E = self._series_set_dilation(X_F, d_rate)
        SX_E = self._series_set_dilation(X_S, d_rate)

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
