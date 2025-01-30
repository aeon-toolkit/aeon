"""ClassifierChannelEnsemble for multivariate time series classification.

Builds classifiers on each channel (dimension) independently.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ClassifierChannelEnsemble"]


import numpy as np
from sklearn.utils import check_random_state

from aeon.base._estimators.compose.collection_channel_ensemble import (
    BaseCollectionChannelEnsemble,
)
from aeon.classification.base import BaseClassifier


class ClassifierChannelEnsemble(BaseCollectionChannelEnsemble, BaseClassifier):
    """Applies estimators to channels of an array.

    Parameters
    ----------
    classifiers : list of aeon and/or sklearn estimators or list of tuples
        Estimators to be used in the ensemble.
        A list of tuples (str, estimator) can also be passed, where the str is used to
        name the estimator.
        The objects are cloned prior. As such, the state of the input will not be
        modified by fitting the ensemble.
    channels : list of int, array-like of int, slice, "all", "all-split" or callable
        Channel(s) to be used with the estimator. Must be the same length as
        ``_estimators``.
        If "all", all channels are used for the estimator. "all-split" will create a
        separate estimator for each channel.
        int, array-like of int and slice are used as indices to select channels. If a
        callable is passed, the input data should return the channel indices to be used.
    remainder : BaseEstimator or None, default=None
        By default, only the specified channels in ``channels`` are
        used and combined in the output, and the non-specified
        channels are dropped.
        By setting `remainder` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support ``fit`` and ``predict``.
    majority_vote : bool, default=False
        If True, the ensemble predictions are the class with the majority of class
        votes from the ensemble.
        If False, the ensemble predictions are the class with the highest probability
        summed from ensemble members.
    random_state : int, RandomState instance or None, default=None
        Random state used to fit the estimators. If None, no random state is set for
        ensemble members (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    ensemble_ : list of tuples (str, estimator) of estimators
        Clones of estimators in classifiers which are fitted in the ensemble.
        Will always be in (str, estimator) format regardless of classifiers input.
    channels_ : list
        The channel indices for each estimator in ``ensemble_``.
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
    }

    def __init__(
        self,
        classifiers,
        channels,
        remainder=None,
        majority_vote=False,
        random_state=None,
    ):
        self.classifiers = classifiers
        self.majority_vote = majority_vote

        super().__init__(
            _ensemble=classifiers,
            channels=channels,
            remainder=remainder,
            random_state=random_state,
            _ensemble_input_name="classifiers",
        )

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X."""
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((len(X), self.n_classes_))

        if self.majority_vote:
            # Call predict on each classifier, add the predictions to the
            # current probabilities
            for i, (_, clf) in enumerate(self.ensemble_):
                preds = clf.predict(X=self._get_channel(X, self.channels_[i]))
                for n in range(X.shape[0]):
                    dists[n, self._class_dictionary[preds[n]]] += 1
        else:
            # Call predict_proba on each classifier, then add them to the current
            # probabilities
            for i, (_, clf) in enumerate(self.ensemble_):
                dists += clf.predict_proba(X=self._get_channel(X, self.channels_[i]))

        # Make each instances probability array sum to 1 and return
        y_proba = dists / dists.sum(axis=1, keepdims=True)

        return y_proba

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            ClassifierChannelEnsemble provides the following special sets:
            - "results_comparison" - used in some classifiers to compare against
              previously generated results where the default set of parameters
              cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from aeon.classification.dictionary_based import ContractableBOSS
        from aeon.classification.interval_based import (
            CanonicalIntervalForestClassifier,
            TimeSeriesForestClassifier,
        )

        if parameter_set == "results_comparison":
            cboss = ContractableBOSS(
                n_parameter_samples=4, max_ensemble_size=2, random_state=0
            )
            cif = CanonicalIntervalForestClassifier(
                n_estimators=2, n_intervals=4, att_subsample_size=4, random_state=0
            )
            return {
                "classifiers": [("cBOSS", cboss), ("CIF", cif)],
                "channels": [5, [3, 4]],
            }
        else:
            return {
                "classifiers": [
                    ("tsf1", TimeSeriesForestClassifier(n_estimators=2)),
                    ("tsf2", TimeSeriesForestClassifier(n_estimators=2)),
                ],
                "channels": [0, 0],
            }
