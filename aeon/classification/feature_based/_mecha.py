"""
The Multiview Enhanced Characteristics (Mecha) classifier.

Mecha is a feature-based Time Series Classification algorithm
with a heterogeneous ensemble structure and an enhancement framework
that includes series shuffling and a Tracking Differentiator
filter factor optimization via Grey Wolf Optimizer.
"""

__maintainer__ = []
__all__ = ["MechaClassifier"]

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.feature_based._mecha_feature_extractor import (
    dilated_fres_extract,
    hard_voting,
    interleaved_fres_extract,
    series_transform,
)
from aeon.utils.validation import check_n_jobs

warnings.filterwarnings("ignore")


def _adaptive_saving_features(
    trainX: np.ndarray, scoresList: list, thresholds: list
) -> np.ndarray:
    """
    Implement the adaptive feature numbers.

    Selection based onstability-diversity scores.
    """
    n_features = trainX.shape[1]
    n_measures = len(scoresList)

    # Use np.nan_to_num to handle potential NaNs before calculating correlation
    temp_trainX = np.nan_to_num(trainX)
    corrM = np.abs(np.corrcoef(temp_trainX, rowvar=False))
    corrM[np.isnan(corrM)] = 0

    stabilityList = np.zeros(n_features)
    diversityList = np.zeros(n_features)
    metricList = np.zeros(n_features)
    sortIndexList = []

    for scores in scoresList:
        sortIndexList.append(np.argsort(scores)[::-1])
    sortIndexList = np.array(sortIndexList)

    selfSum_ = 0
    mutualSum_ = 0

    for i in range(n_features):
        saveN = i + 1

        # Calculate Diversity Score
        diversitySet_ = []
        for j in range(n_measures):
            sortIndex_ = sortIndexList[j, :saveN]
            indexlLast_ = sortIndex_[-1]
            newSubM = corrM[indexlLast_, sortIndex_]
            selfSum_ += 2 * np.sum(newSubM) - 1

        diversitySet_.append(selfSum_ / (n_measures * saveN**2))

        # Calculate Stability Score
        stabilitySet_ = []
        for j in range(n_measures):
            for k in range(j + 1, n_measures):
                sortIndex0_ = sortIndexList[j, :saveN]
                sortIndex1_ = sortIndexList[k, :saveN]
                indexLast0_ = sortIndex0_[-1]
                indexLast1_ = sortIndex1_[-1]

                newSubM0 = corrM[indexLast1_, sortIndex0_]
                newSubM1 = corrM[indexLast0_, sortIndex1_]

                mutualSum_ += (
                    np.sum(newSubM0)
                    + np.sum(newSubM1)
                    - corrM[indexLast0_, indexLast1_]
                )

        stabilitySet_.append(
            mutualSum_ / ((n_measures * (n_measures - 1) // 2) * (saveN**2))
        )

        stabilityList[i] = np.mean(stabilitySet_)
        diversityList[i] = np.mean(diversitySet_)

        if diversityList[i] == 0:
            metricList[i] = np.inf
        else:
            metricList[i] = stabilityList[i] / diversityList[i]

    metric_diff_rate = np.abs(np.diff(metricList) / metricList[:-1])
    metric_diff_rate = metric_diff_rate[::-1]
    kList = np.ones(len(thresholds), int) * len(metric_diff_rate)

    for k in range(len(thresholds)):
        for i in range(len(metric_diff_rate)):
            if metric_diff_rate[i] <= thresholds[k]:
                kList[k] -= 1
            else:
                break

    return kList + 1


def _objective_function(
    trainSeriesX: np.ndarray, trainY: np.ndarray, k1: np.ndarray, down_rate: int
) -> float:
    """
    Objective function for the Grey Wolf Optimizer - Maximizes Silhouette Score.

    k1 is passed as a 1D array of size dim=1 from the GWO.
    """
    trainSeriesFX = series_transform(trainSeriesX, k1=k1[0])
    n_cases = trainSeriesFX.shape[0]
    trainSeriesFX = trainSeriesFX.reshape([n_cases, -1])

    trainX = np.sort(trainSeriesFX, axis=1)[:, ::down_rate]

    if trainX.shape[0] < 2 or trainX.shape[1] == 0:
        return -np.inf

    scaler = MinMaxScaler()
    trainX_scaled = scaler.fit_transform(trainX)

    unique_y = np.unique(trainY)
    if len(unique_y) < 2:
        return -np.inf

    # Silhouette score is the metric used for clustering quality/compactness
    score = silhouette_score(trainX_scaled, trainY)

    return score


def _gwo(
    objective_function: callable,
    trainSeriesX: np.ndarray,
    trainY: np.ndarray,
    dim: int,
    search_space: list,
    down_rate: int,
    num_wolves: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Implement the Gray Wolf Optimizer."""
    np.random.seed(seed)
    wolves = np.random.uniform(
        low=search_space[0], high=search_space[1], size=(num_wolves, dim)
    )

    alpha_pos = np.zeros(dim)
    alpha_score = -np.inf

    beta_pos = np.zeros(dim)
    beta_score = -np.inf

    delta_pos = np.zeros(dim)
    delta_score = -np.inf

    for t in range(max_iter):
        for i in range(num_wolves):
            # Pass the wolves position array (k1) to the objective function
            fitness = objective_function(
                trainSeriesX, trainY, k1=wolves[i], down_rate=down_rate
            )

            # Update the alpha, beta, and delta positions (tracking the best scores)
            if fitness > alpha_score:
                delta_pos = beta_pos.copy()
                delta_score = beta_score
                beta_pos = alpha_pos.copy()
                beta_score = alpha_score
                alpha_pos = wolves[i].copy()
                alpha_score = fitness
            elif fitness > beta_score:
                delta_pos = beta_pos.copy()
                delta_score = beta_score
                beta_pos = wolves[i].copy()
                beta_score = fitness
            elif fitness > delta_score:
                delta_pos = wolves[i].copy()
                delta_score = fitness

        # Update the wolves' positions
        a = 2 - t * (2 / max_iter)

        for i in range(num_wolves):
            # Use seed + i + t * num_wolves for non-overlapping random sequences
            np.random.seed(seed + i + t * num_wolves)
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A = 2 * a * r1 - a
            C = 2 * r2

            D_alpha = abs(C * alpha_pos - wolves[i])
            D_beta = abs(C * beta_pos - wolves[i])
            D_delta = abs(C * delta_pos - wolves[i])

            X1 = alpha_pos - A * D_alpha
            X2 = beta_pos - A * D_beta
            X3 = delta_pos - A * D_delta

            wolves[i] = (X1 + X2 + X3) / 3

            # Clip the position to the search space boundaries
            wolves[i] = np.clip(wolves[i], search_space[0], search_space[1])

    return alpha_pos, alpha_score


class MechaClassifier(BaseClassifier):
    """
    Multiview Enhanced Characteristics (Mecha) for Time Series Classification.

    Mecha uses a diverse feature extractor (TD and Series Shuffling), an adaptive
    feature selector based on stability/diversity scores, and a heterogeneous
    ensemble of Ridge Regression and Extremely Randomized Trees classifiers.

    Parameters
    ----------
    basic_extractor : str, default="Catch22"
        Basic feature extractor. Options are "Catch22", or "TSFresh".
        (Note: TSFresh currently use efficient feature set.)
    search_space : list, default=[1.0, 3.0]
        The boundaries for the filter factor of the TD during GWO optimization.
    down_rate : int, default=4
        The downsampling rate applied after sorting features in the GWO objective.
    num_wolves : int, default=10
        The number of wolves (agents) in the GWO.
    max_iter : int, default=10
        The maximum iteration in GWO.
    max_rate : int, default=16
        Maximum shuffling rate for dilation and interleaving mappings.
    thresholds : list, default=[2e-05, 4e-05, 6e-05, 8e-05, 10e-05]
        Convergence thresholds used in the adaptive feature selector.
    n_trees : int, default=200
        The number of trees in the ExtraTrees classifier ensemble members.
    random_state: int, default=0
        Controls randomness in GWO, ExtraTrees, and feature selection.

    References
    ----------
    .. [1] Changchun He, Xin Huo, Baohan Mi, and Songlin Chen. "Mecha: Multiview
       Enhanced Characteristics via Series Shuffling for Time Series Classification
       and Its Application to Turntable circuit", IEEE Transactions on Circuits
       and Systems I: Regular Papers, 2025.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "feature",
        "X_inner_type": ["numpy3D"],
    }

    def __init__(
        self,
        basic_extractor="Catch22",
        search_space=None,
        down_rate=4,
        num_wolves=10,
        max_iter=10,
        max_rate=16,
        thresholds=None,
        n_trees=200,
        random_state=0,
        n_jobs=1,
    ) -> None:
        self.search_space = search_space
        self.thresholds = thresholds
        self.basic_extractor = basic_extractor
        self.down_rate = down_rate
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.max_rate = max_rate
        self.n_trees = n_trees
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._n_jobs = check_n_jobs(self.n_jobs)

        supported_extractors = ["Catch22", "TSFresh", "TSFreshRelevant"]
        if self.basic_extractor not in supported_extractors:
            raise ValueError(
                f"basic_extractor must be one of {supported_extractors}. Found: "
                f"{self.basic_extractor}"
            )

        # Internal fitted attributes
        self.optimized_k1 = None
        self.optimized_score = None
        self.scaler = None
        self.indexListMI = None
        self.indexListFV = None
        self.indexListA = None
        self.clfListRidgeMI = None
        self.clfListExtraMI = None
        self.clfListRidgeFV = None
        self.clfListExtraFV = None
        self.clfListRidgeA = None
        self.clfListExtraA = None

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Mecha classifier on the training data (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The training input samples.
        y : array-like, shape = (n_cases,)
            The class labels.

        Returns
        -------
        self : object
            Reference to self.
        """
        search_space_ = (
            self.search_space if self.search_space is not None else [1.0, 3.0]
        )
        thresholds_ = (
            self.thresholds
            if self.thresholds is not None
            else [2e-05, 4e-05, 6e-05, 8e-05, 10e-05]
        )
        # 1. Series Transformation & GWO Optimization
        self.optimized_k1, self.optimized_score = _gwo(
            _objective_function,
            X,
            y,
            dim=1,
            search_space=search_space_,
            down_rate=self.down_rate,
            num_wolves=self.num_wolves,
            max_iter=self.max_iter,
            seed=self.random_state,
        )
        trainSeriesFX = series_transform(X, k1=self.optimized_k1[0])

        # 2. Diverse Feature Extraction
        trainRX_Drie = dilated_fres_extract(
            X, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        trainFX_Drie = dilated_fres_extract(
            trainSeriesFX, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        trainRX_Inve = interleaved_fres_extract(
            X, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        trainFX_Inve = interleaved_fres_extract(
            trainSeriesFX, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )

        trainX = np.hstack((trainRX_Drie, trainRX_Inve, trainFX_Drie, trainFX_Inve))

        # 3. Feature Normalization
        self.scaler = MinMaxScaler()
        self.scaler.fit(trainX)
        trainX = self.scaler.transform(trainX)

        # 4. Ensemble Feature Selection
        scoreList = []
        scoreMI = mutual_info_classif(trainX, y, random_state=self.random_state)
        scoreMI[np.isnan(scoreMI)] = 0
        scoreMI[np.isinf(scoreMI)] = 0
        scoreList.append(scoreMI)

        scoreFV = f_classif(trainX, y)[0]
        scoreFV[np.isnan(scoreFV)] = 0
        scoreFV[np.isinf(scoreFV)] = 0
        scoreList.append(scoreFV)

        kList = _adaptive_saving_features(trainX, scoreList, thresholds_)

        self.indexListMI = []
        self.indexListFV = []
        self.indexListA = []

        for bestN in kList:
            bestN = np.max([100, bestN])

            indexMI = np.argsort(scoreList[0])[::-1][:bestN]
            indexFV = np.argsort(scoreList[1])[::-1][:bestN]

            indexA = np.intersect1d(indexMI, indexFV)
            if len(indexA) == 0:
                indexA = np.hstack((indexMI[: bestN // 2], indexFV[: bestN // 2]))

            self.indexListMI.append(indexMI)
            self.indexListFV.append(indexFV)
            self.indexListA.append(indexA)

        # 5. Heterogeneous Ensemble Classifier Training
        self.clfListRidgeMI, self.clfListExtraMI = [], []
        self.clfListRidgeFV, self.clfListExtraFV = [], []
        self.clfListRidgeA, self.clfListExtraA = [], []

        for i in range(len(self.indexListMI)):
            # MI View
            bestIndex_ = self.indexListMI[i]
            clf_ridge_mi = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf_extra_mi = ExtraTreesClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            clf_ridge_mi.fit(trainX[:, bestIndex_], y)
            clf_extra_mi.fit(trainX[:, bestIndex_], y)
            self.clfListRidgeMI.append(clf_ridge_mi)
            self.clfListExtraMI.append(clf_extra_mi)

            # FV View
            bestIndex_ = self.indexListFV[i]
            clf_ridge_fv = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf_extra_fv = ExtraTreesClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            clf_ridge_fv.fit(trainX[:, bestIndex_], y)
            clf_extra_fv.fit(trainX[:, bestIndex_], y)
            self.clfListRidgeFV.append(clf_ridge_fv)
            self.clfListExtraFV.append(clf_extra_fv)

            # Intersection View
            bestIndex_ = self.indexListA[i]
            clf_ridge_a = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf_extra_a = ExtraTreesClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            clf_ridge_a.fit(trainX[:, bestIndex_], y)
            clf_extra_a.fit(trainX[:, bestIndex_], y)
            self.clfListRidgeA.append(clf_ridge_a)
            self.clfListExtraA.append(clf_extra_a)

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class values for n instances in X.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = (n_cases)
            Predicted class labels.
        """
        # 1. Series Transformation

        if X is None or len(X) == 0:
            raise ValueError("Input data X cannot be empty.")

        testSeriesFX = series_transform(X, k1=self.optimized_k1[0])

        # 2. Diverse Feature Extraction
        testRX_Drie = dilated_fres_extract(
            X, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testFX_Drie = dilated_fres_extract(
            testSeriesFX, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testRX_Inve = interleaved_fres_extract(
            X, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testFX_Inve = interleaved_fres_extract(
            testSeriesFX, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testX = np.hstack((testRX_Drie, testRX_Inve, testFX_Drie, testFX_Inve))

        # 3. Feature Normalization
        testX = self.scaler.transform(testX)

        # 4. Heterogeneous Ensemble Prediction (Hard Voting)
        testPYListMI_RL, testPYListFV_RL, testPYListA_RL = [], [], []
        testPYListMI_ET, testPYListFV_ET, testPYListA_ET = [], [], []

        for i in range(len(self.indexListMI)):
            # MI View
            bestIndex_ = self.indexListMI[i]
            clf_ridge_mi = self.clfListRidgeMI[i]
            clf_extra_mi = self.clfListExtraMI[i]
            testPY_Doub0 = clf_ridge_mi.predict(testX[:, bestIndex_])
            testPY_Doub1 = clf_extra_mi.predict(testX[:, bestIndex_])
            testPYListMI_RL.append(testPY_Doub0)
            testPYListMI_ET.append(testPY_Doub1)

            # FV View
            bestIndex_ = self.indexListFV[i]
            clf_ridge_fv = self.clfListRidgeFV[i]
            clf_extra_fv = self.clfListExtraFV[i]
            testPY_Doub0 = clf_ridge_fv.predict(testX[:, bestIndex_])
            testPY_Doub1 = clf_extra_fv.predict(testX[:, bestIndex_])
            testPYListFV_RL.append(testPY_Doub0)
            testPYListFV_ET.append(testPY_Doub1)

            # Intersection View
            bestIndex_ = self.indexListA[i]
            clf_ridge_a = self.clfListRidgeA[i]
            clf_extra_a = self.clfListExtraA[i]
            testPY_Doub0 = clf_ridge_a.predict(testX[:, bestIndex_])
            testPY_Doub1 = clf_extra_a.predict(testX[:, bestIndex_])
            testPYListA_RL.append(testPY_Doub0)
            testPYListA_ET.append(testPY_Doub1)

        testPYListMI_RL = np.array(testPYListMI_RL)
        testPYListFV_RL = np.array(testPYListFV_RL)
        testPYListA_RL = np.array(testPYListA_RL)
        testPYListMI_ET = np.array(testPYListMI_ET)
        testPYListFV_ET = np.array(testPYListFV_ET)
        testPYListA_ET = np.array(testPYListA_ET)

        # Final Hard Voting across all classifiers
        testPY = hard_voting(
            np.vstack(
                (
                    testPYListMI_RL,
                    testPYListFV_RL,
                    testPYListA_RL,
                    testPYListMI_ET,
                    testPYListFV_ET,
                    testPYListA_ET,
                )
            )
        )

        return testPY

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for n instances in X.

        Mecha uses hard voting on labels, but we provide an averaged probability
        estimate from the ExtraTrees sub-classifiers for API completeness.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = (n_cases, n_classes_)
            Predicted probabilities.
        """
        # 1. Feature Extraction & Normalization
        testSeriesFX = series_transform(X, k1=self.optimized_k1[0])
        testRX_Drie = dilated_fres_extract(
            X, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testFX_Drie = dilated_fres_extract(
            testSeriesFX, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testRX_Inve = interleaved_fres_extract(
            X, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testFX_Inve = interleaved_fres_extract(
            testSeriesFX, max_rate=self.max_rate, basic_extractor=self.basic_extractor
        )
        testX = np.hstack((testRX_Drie, testRX_Inve, testFX_Drie, testFX_Inve))
        testX = self.scaler.transform(testX)

        # 2. Ensemble Probability Prediction (only ExtraTrees)
        probas = []

        for i in range(len(self.indexListMI)):
            # MI View
            clf_extra_mi = self.clfListExtraMI[i]
            probas.append(clf_extra_mi.predict_proba(testX[:, self.indexListMI[i]]))

            # FV View
            clf_extra_fv = self.clfListExtraFV[i]
            probas.append(clf_extra_fv.predict_proba(testX[:, self.indexListFV[i]]))

            # Intersection View
            clf_extra_a = self.clfListExtraA[i]
            probas.append(clf_extra_a.predict_proba(testX[:, self.indexListA[i]]))

        return np.mean(probas, axis=0)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params_c22 = {
            "max_iter": 1,
            "num_wolves": 2,
            "max_rate": 2,
            "basic_extractor": "Catch22",
            "n_jobs": 1,
        }

        params_tsf = {
            "max_iter": 1,
            "num_wolves": 2,
            "max_rate": 2,
            "basic_extractor": "TSFresh",
            "n_jobs": 1,
        }

        return [params_c22, params_tsf]
