import numpy as np
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

__all__ = ["TDMVDCClassifier"]


class TDMVDCClassifier(BaseClassifier):
    """
    Tracking Differentiator-based Multiview Dilated Characteristics.

    for Time Series Classification.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    k1 : floot, default=2
        Filter parameter of the Tracking Differentiator1 with generating first-order
        differential series
    k2 : floot, default=2
        Filter parameter of the Tracking Differentiator2 with generating second-order
        differential series
    feature_store_ratios : list, default=[0.1, 0.2, 0.3, 0.4, 0.5]
        List of feature saving ratios for different feature selectors
    n_jobs : int, default=1
        The number of jobs to run in parallel for both fit and predict.


    References
    ----------
    .. [1] Changchun He, and Xin Huo. "Tracking Differentiator-based Multiview Dilated
        Characteristics for Time Series Classification." in The 22nd IEEE International
        Conference on Industrial Informatics (INDIN2024) (2024).
    """

    _tags = {
        "capability:multivariate": False,
        "capability:multithreading": False,
        "capability:train_estimate": False,
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
    ):
        self.default_fc_parameters = default_fc_parameters
        self.k1 = k1
        self.k2 = k2
        if feature_store_ratios is None:
            feature_store_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.feature_store_ratios = feature_store_ratios
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, trainSignalX, trainY):
        """Fit a pipeline on cases (trainSignalX, trainY).

        where trainY is the target variable.

        Parameters
        ----------
        trainSignalX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        trainY : array-like, shape = [n_cases]
            The class labels. Each type of label is int.

        Returns
        -------
        self :
            Reference to self.
        """
        # Initialization of dilation rate parameters
        n_timepoints = trainSignalX.shape[2]  # The number of time points
        d_min = 0  # The minimum dilation rate corresponds to no dilation
        d_max = int(np.log2(n_timepoints - 1) - 3)
        d_max = np.min([5, d_max])  # The maximum dilation rate
        self.dList = 2 ** np.arange(d_min, d_max + 1)  # The dilation rate list

        # Differential transformations by tracking differentiator
        trainSignalFX = series_transform(
            trainSignalX, mode=1, k1=self.k1
        )  # First-order differential
        trainSignalSX = series_transform(
            trainSignalX, mode=2, k1=self.k1, k2=self.k2
        )  # Second-order differential

        # Feature extraction
        self.tsFreshListR = []
        # List of feature extractors corresponding
        # to the original series set
        self.tsFreshListF = []
        # List of feature extractors corresponding
        # to the First-order differential series set
        self.tsFreshListS = []
        # List of feature extractors corresponding
        # to the Second-order differential series set

        trainRXList = []
        # List of train feature sets corresponding
        # to the original series set
        trainFXList = []
        # List of train feature sets corresponding
        # to the First-order differential series set
        trainSXList = []
        # List of train feature sets corresponding
        # to the Second-order differential series set

        for i in range(len(self.dList)):  # For each dilation rate
            # Dilation Mapping
            d_rate = self.dList[i]  # Dilation rate
            trainSignalRX_E = series_set_dilation(
                trainSignalX, d_rate
            )  # Dilated original series set
            trainSignalFX_E = series_set_dilation(
                trainSignalFX, d_rate
            )  # Dilated First-order differential series set
            trainSignalSX_E = series_set_dilation(
                trainSignalSX, d_rate
            )  # Dilated Second-order differential series set

            # Extracting the TSFresh features for each dilated series set
            tsFreshR = TSFresh(default_fc_parameters="efficient", n_jobs=self.n_jobs)
            tsFreshR.fit(trainSignalRX_E, trainY)
            trainRX = np.array(tsFreshR.transform(trainSignalRX_E))

            tsFreshF = TSFresh(default_fc_parameters="efficient", n_jobs=self.n_jobs)
            tsFreshF.fit(trainSignalFX_E, trainY)
            trainFX = np.array(tsFreshF.transform(trainSignalFX_E))

            tsFreshS = TSFresh(default_fc_parameters="efficient", n_jobs=self.n_jobs)
            tsFreshS.fit(trainSignalSX_E, trainY)
            trainSX = np.array(tsFreshS.transform(trainSignalSX_E))

            # Saving the feature extractors
            self.tsFreshListR.append(tsFreshR)
            self.tsFreshListF.append(tsFreshF)
            self.tsFreshListS.append(tsFreshS)

            # Saving the TSFresh features
            trainRXList.append(trainRX)
            trainFXList.append(trainFX)
            trainSXList.append(trainSX)

        # Concatenating all the dilated features into a feature set
        trainRX = np.hstack(trainRXList)  # Corresponding to the original series set
        trainFX = np.hstack(
            trainFXList
        )  # Corresponding to the First-order differential series set
        trainSX = np.hstack(
            trainSXList
        )  # Corresponding to the Second-order differential series set

        # Classification
        self.clfList = []  # List of classifiers composed of different views

        # Computing feature scores
        self.scoreRFS = f_classif(np.hstack((trainRX, trainFX, trainSX)), trainY)[0]
        self.scoreRFS[np.isnan(self.scoreRFS)] = 0

        # Training the classifier on each view
        for i in range(len(self.feature_store_ratios)):  # for each view
            ratio_ = self.feature_store_ratios[i]  # The feature store ratio
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),  # Normalize the features
                    ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
                ]
            )
            bestIndex_ = np.argsort(self.scoreRFS)[::-1][
                0 : int(len(self.scoreRFS) * ratio_)
            ]  # The feature indexes of the top scores
            clf.fit(np.hstack((trainRX, trainFX, trainSX))[:, bestIndex_], trainY)
            self.clfList.append(clf)  # Saving the trained classifier
        return self

    def _predict(self, testSignalX):
        """Predict class values of n instances in testSignalX.

        Parameters
        ----------
        testSignalX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for testSignalX.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        testSignalFX = series_transform(
            testSignalX, mode=1, k1=self.k1
        )  # First-order differential
        testSignalSX = series_transform(
            testSignalX, mode=2, k1=self.k1, k2=self.k2
        )  # Second-order differential

        # Feature extraction
        testRXList = []
        # List of test feature sets corresponding
        # to the original series set
        testFXList = []
        # List of test feature sets corresponding
        # to the First-order differential series set
        testSXList = []
        # List of test feature sets corresponding
        # to the Second-order differential series set

        for i in range(len(self.dList)):  # For each dilation rate
            # Dilation Mapping
            d_rate = self.dList[i]  # Dilation rate
            testSignalRX_E = series_set_dilation(
                testSignalX, d_rate
            )  # Dilated original series set
            testSignalFX_E = series_set_dilation(
                testSignalFX, d_rate
            )  # Dilated First-order differential series set
            testSignalSX_E = series_set_dilation(
                testSignalSX, d_rate
            )  # Dilated Second-order differential series set

            # Extracting the TSFresh features for each dilated series set
            tsFreshR = self.tsFreshListR[i]
            testRX = np.array(tsFreshR.transform(testSignalRX_E))

            tsFreshF = self.tsFreshListF[i]
            testFX = np.array(tsFreshF.transform(testSignalFX_E))

            tsFreshS = self.tsFreshListS[i]
            testSX = np.array(tsFreshS.transform(testSignalSX_E))

            # Saving the TSFresh features
            testRXList.append(testRX)
            testFXList.append(testFX)
            testSXList.append(testSX)

        # Concatenating all the dilated features into a feature set
        testRX = np.hstack(testRXList)  # Corresponding to the original series set
        testFX = np.hstack(
            testFXList
        )  # Corresponding to the First-order differential series set
        testSX = np.hstack(
            testSXList
        )  # Corresponding to the Second-order differential series set

        # Classification
        testPYList = []  # List of predicted labels on each view
        # Predicting the labels of each view
        for i in range(len(self.feature_store_ratios)):  # for each view
            ratio_ = self.feature_store_ratios[i]  # The feature store ratio
            clf = self.clfList[i]
            bestIndex_ = np.argsort(self.scoreRFS)[::-1][
                0 : int(len(self.scoreRFS) * ratio_)
            ]
            testPY_ = clf.predict(
                np.hstack((testRX, testFX, testSX))[:, bestIndex_]
            )  # prediction
            testPYList.append(testPY_)  # Saving the predicted labels
        testPYV = hard_voting(
            np.vstack(testPYList)
        )  # The final predicted labels is generated by hard voting
        return testPYV

    def _predict_proba(self, X):  # optional
        return super()._predict_proba(X)

    def _fit_predict(self, X, y) -> np.ndarray:
        pass

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        pass
