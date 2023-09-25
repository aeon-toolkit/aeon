# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

"""Random EnhanceD Co-eye for Multivariate Time Series (RED CoMETS).

Ensemble of symbolically represented time series using random forests as the base
classifier.
"""

__author__ = ["zy18811"]
__all__ = ["REDCOMETS"]

from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.dictionary_based import SAX, SFA
from aeon.utils.validation._dependencies import _check_soft_dependencies


class REDCOMETS(BaseClassifier):
    """
    Random EnhanceD Co-eye for Multivariate Time Series (RED CoMETS).

    Ensemble of symbolically represented time series using random forests as the base
    classifier as described in [1]. Based on Co-eye [2].

    Parameters
    ----------
    variant : int, default=3
        RED CoMETS variant to use as per [1]. Defaults to RED CoMETS-3.
    perc_length : int, default=5
        Percentage of time series length used to determinne number of lenses during
        pair selection.
    n_trees : int, default=100
        Number of trees used by each random forest sub-classifier.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.

    Notes
    -----
    Adapted from the implementation at https://github.com/zy18811/RED-CoMETS

    References
    ----------
    .. [1] Luca A. Bennett and Zahraa S. Abdallah, "RED CoMETS: An ensemble classifier
       for symbolically represented multivariate time series."
       Preprint, https://arxiv.org/abs/2307.13679
    .. [2] Zahraa S. Abdallah and Mohamed Medhat Gaber, "Co-eye: a multi-resolution
       ensemble classifier for symbolically approximated time series."
       Machine Learning (2020).
    """

    _tags = {
        "python_dependencies": "imblearn",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        variant=3,
        perc_length=5,
        n_trees=100,
        random_state=None,
        n_jobs=1,
    ):
        _check_soft_dependencies(
            "imbalanced-learn",
            package_import_alias={"imbalanced-learn": "imblearn"},
            severity="error",
            obj=self,
        )

        self.variant = variant
        self.perc_length = perc_length
        self.n_trees = n_trees

        self.random_state = random_state
        self.n_jobs = n_jobs

        self._n_channels = 1

        self.sfa_clfs = []
        self.sfa_transforms = []

        self.sax_clfs = []
        self.sax_transforms = []

        super(REDCOMETS, self).__init__()

    def _fit(self, X, y):
        """Build a REDCOMETS classifier from the training set (X, y).

        Parameters
        ----------
        X : 3D np.ndarray
            The training data shape = (n_instances, n_channels, n_timepoints).
        y : 1D np.ndarray
            The class labels shape = (n_instances).

        Returns
        -------
        self :
            Reference to self.
        """
        if (n_channels := X.shape[1]) == 1:  # Univariate
            self._fit_univariate(np.squeeze(X), y)
        else:  # Multivariate
            if self.variant in [1, 2, 3]:  # Concatenate dimensions
                self._n_channels = n_channels
                X_concat = X.reshape(*X.shape[:-2], -1)
                self._fit_univariate(X_concat, y)

            elif self.variant in [4, 5, 6, 7, 8, 9]:  # Ensemble over dimensions
                pass

    def _fit_univariate(self, X, y):
        """Build a univariate REDCOMETS classifier from the training set (X, y).

        Parameters
        ----------
        X : 3D np.ndarray
            The training data shape = (n_instances, n_channels, n_timepoints).
        y : 1D np.ndarray
            The class labels shape = (n_instances).

        Returns
        -------
        self :
            Reference to self.
        """
        from imblearn.over_sampling import SMOTE, RandomOverSampler

        perc_length = self.perc_length / self._n_channels
        n_lenses = 2 * int(perc_length * X.shape[1] // 100)

        min_neighbours = min(Counter(y).items(), key=lambda k: k[1])[1]
        max_neighbours = max(Counter(y).items(), key=lambda k: k[1])[1]

        if min_neighbours == max_neighbours:
            X_smote = X
            y_smote = y

        else:
            if min_neighbours > 5:
                min_neighbours = 6
            try:
                X_smote, y_smote = SMOTE(
                    sampling_strategy="all",
                    k_neighbors=NearestNeighbors(
                        n_neighbors=min_neighbours - 1, n_jobs=self.n_jobs
                    ),
                    random_state=self.random_state,
                ).fit_resample(X, y)

            except ValueError:
                X_smote, y_smote = RandomOverSampler(
                    sampling_strategy="all", random_state=self.random_state
                ).fit_resample(X, y)

        sax_lenses, sfa_lenses = np.split(self._get_random_lenses(X_smote, n_lenses), 2)

        cv = np.min([5, len(y_smote) // len(list(set(y_smote)))])

        self.sfa_transforms = [
            SFA(
                word_length=w,
                alphabet_size=a,
                window_size=X_smote.shape[1],
                binning_method="equi-width",
                save_words=True,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            for w, a in sfa_lenses
        ]

        for sfa in self.sfa_transforms:
            sfa.fit_transform(X_smote, y_smote)
            X_sfa = np.array(
                list(map(lambda word: sfa.word_list(int(word[0])), sfa.words))
            )

            rf = RandomForestClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            rf.fit(X_sfa, y_smote)
            weight = cross_val_score(
                rf, X_sfa, y_smote, cv=cv, n_jobs=self.n_jobs
            ).mean()

            self.sfa_clfs.append((rf, weight))

        self.sax_transforms = [
            SAX(n_segments=w, alphabet_size=a) for w, a in sax_lenses
        ]

        for sax in self.sax_transforms:
            X_sax = np.squeeze(sax.fit_transform(X_smote))

            rf = RandomForestClassifier(
                n_estimators=self.n_trees,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            rf.fit(X_sax, y_smote)
            weight = cross_val_score(
                rf, X_sax, y_smote, cv=cv, n_jobs=self.n_jobs
            ).mean()

            self.sax_clfs.append((rf, weight))

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return np.array(
            [self.classes_[i] for i in self._predict_proba(X).argmax(axis=1)]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        if X.shape[1] == 1:  # Univariate
            return self._predict_proba_unvivariate(np.squeeze(X))
        else:  # Multivariate
            if self.variant in [1, 2, 3]:  # Concatenate dimensions
                X_concat = X.reshape(*X.shape[:-2], -1)
                return self._predict_proba_unvivariate(X_concat)

    def _predict_proba_unvivariate(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in univariate X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        pred_mat = np.zeros((X.shape[0], self.n_classes_))

        placeholder_y = np.zeros(X.shape[0])
        for sfa, (rf, weight) in zip(self.sfa_transforms, self.sfa_clfs):
            sfa.fit_transform(X, placeholder_y)
            X_sfa = np.array(
                list(map(lambda word: sfa.word_list(int(word[0])), sfa.words))
            )

            pred_mat += rf.predict_proba(X_sfa) * weight

        for sax, (rf, weight) in zip(self.sax_transforms, self.sax_clfs):
            X_sax = np.squeeze(sax.fit_transform(X))

            pred_mat += rf.predict_proba(X_sax) * weight

        pred_mat /= np.sum(pred_mat, axis=1).reshape(-1, 1)  # Rescales rows to sum to 1

        return pred_mat

    def _get_random_lenses(self, X, n_lenses):
        """Randomly select <word length, alphabet size> pairs.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data shape = (n_instances, n_channels, n_timepoints).
        n_lenses : int
            Number of lenses to select.

        Returns
        -------
        lenses : array-like, shape = [n_lenses, 2]
            Selected lenses.
        """
        maxCoof = 130

        if X.shape[1] < maxCoof:
            maxCoof = X.shape[1] - 1
        if X.shape[1] < 100:
            n_segments = list(range(5, maxCoof, 5))
        else:
            n_segments = list(range(10, maxCoof, 10))

        maxBin = 26
        if X.shape[1] < maxBin:
            maxBin = X.shape[1] - 2
        if X.shape[0] < maxBin:
            maxBin = X.shape[0] - 2

        alphas = range(3, maxBin)

        rng = check_random_state(self.random_state)
        lenses = np.array(
            list(
                zip(
                    rng.choice(n_segments, size=n_lenses),
                    rng.choice(alphas, size=n_lenses),
                )
            )
        )
        return lenses
