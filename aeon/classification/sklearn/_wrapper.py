"""Wrapper for scikit-learn classifiers to use the aeon framework."""

from sklearn.ensemble import RandomForestClassifier

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier


class SklearnClassifierWrapper(BaseClassifier):
    """Wrapper for scikit-learn classifiers to use the aeon framework.

    Parameters
    ----------
    classifier : sklearn BaseEstimator
        A scikit-learn classifier object.
    random_state : int, RandomState instance or None, default=None
        Random state set when cloning the estimator. If None, no random
        state is set (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    classifier_ : object
        The cloned scikit-learn classifier object.
    """

    _tags = {
        "X_inner_type": "numpy2D",
    }

    def __init__(self, classifier, random_state=None):
        self.classifier = classifier
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self.classifier_ = _clone_estimator(self.classifier, self.random_state)
        self.classifier_.fit(X, y)
        return self

    def _predict(self, X):
        return self.classifier_.predict(X)

    def _predict_proba(self, X):
        return self.classifier_.predict_proba(X)

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "classifier": RandomForestClassifier(n_estimators=5),
        }
