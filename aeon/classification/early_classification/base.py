"""
Abstract base class for early time series classifiers.

    class name: BaseEarlyClassifier

Defining methods:
    fitting                 - fit(self, X, y)
    predicting              - predict(self, X)
                            - predict_proba(self, X)
    updating predictions    - update_predict(self, X)
      (streaming)           - update_predict_proba(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
    streaming decision info - state_info attribute
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseEarlyClassifier"]

from abc import abstractmethod

import numpy as np

from aeon.base import BaseCollectionEstimator
from aeon.classification import BaseClassifier


class BaseEarlyClassifier(BaseCollectionEstimator):
    """
    Abstract base class for early time series classifiers.

    The base classifier specifies the methods and method signatures that all
    early classifiers have to implement. Attributes with an underscore suffix are set in
    the method fit.

    Parameters
    ----------
    classes_ : np.ndarray
        Class labels, possibly strings.
    n_classes_ : int
        Number of classes (length of classes_).
    _class_dictionary : dict
        dictionary mapping classes_ onto integers 0...n_classes_-1.
    _n_jobs : int, default=1
        Number of threads to use in fit as determined by n_jobs.
    state_info : array-like, default=None
        An array containing the state info for each decision in X.
    """

    _tags = {
        "fit_is_empty": False,
    }

    @abstractmethod
    def __init__(self):
        self.classes_ = []
        self.n_classes_ = 0
        self._class_dictionary = {}

        """
        An array containing the state info for each decision in X from update and
        predict methods. Contains classifier dependant information for future decisions
        on the data and information on when a cases decision has been made. Each row
        contains information for a case from the latest decision on its safety made in
        update/predict. Successive updates are likely to remove rows from the
        state_info, as it will only store as many rows as there are input instances to
        update/predict.
        """
        self.state_info = None

        super().__init__()

    def fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. Other types are
            allowed and converted into one of the above.
        np.array
            shape ``(n_cases)`` - class labels for fitting indices correspond to
            instance indices in X.

        Returns
        -------
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        # reset estimator at the start of fit
        self.reset()

        # All of this can move up to BaseCollection
        X = self._preprocess_collection(X)
        y = BaseClassifier._check_y(self, y, self.metadata_["n_cases"])
        self._fit(X, y)
        # this should happen last
        self.is_fitted = True
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predicts labels for sequences in X.

        Early classifiers can predict at series lengths shorter than the train data
        series length.

        Predict will return -1 for cases which it cannot make a decision on yet. The
        output is only guaranteed to return a valid class label for all cases when
        using the full series length.

        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``
            other types are allowed and converted into one of the above.

        Returns
        -------
        y : np.array
            shape ``[n_cases]`` - predicted class labels indices correspond to
            instance indices in X.
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use.
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X)
        return self._predict(X)

    def update_predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Update label prediction for sequences in X at a larger series length.

        Uses information stored in the classifiers state from previous predictions and
        updates at shorter series lengths. Update will only accept cases which have not
        yet had a decision made, cases which have had a positive decision should be
        removed from the input with the row ordering preserved.

        If no state information is present, predict will be called instead.

        Prediction updates will return -1 for cases which it cannot make a decision on
        yet. The output is only guaranteed to return a valid class label for all cases
        when using the full series length.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 1D np.array of int, of shape [n_cases] - predicted class labels
            indices correspond to instance indices in X
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        self._check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._preprocess_collection(X)

        if self.state_info is None:
            return self._predict(X)
        else:
            return self._update_predict(X)

    def predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predicts labels probabilities for sequences in X.

        Early classifiers can predict at series lengths shorter than the train data
        series length.

        Probability predictions will return [-1]*n_classes_ for cases which it cannot
        make a decision on yet. The output is only guaranteed to return a valid class
        label for all cases when using the full series length.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X)

        return self._predict_proba(X)

    def update_predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Update label probabilities for sequences in X at a larger series length.

        Uses information stored in the classifiers state from previous predictions and
        updates at shorter series lengths. Update will only accept cases which have not
        yet had a decision made, cases which have had a positive decision should be
        removed from the input with the row ordering preserved.

        If no state information is present, predict_proba will be called instead.

        Probability predictions updates will return [-1]*n_classes_ for cases which it
        cannot make a decision on yet. The output is only guaranteed to return a valid
        class label for all cases when using the full series length.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X)
        if self.state_info is None:
            return self._predict_proba(X)
        else:
            return self._update_predict_proba(X)

    def score(self, X, y) -> tuple[float, float, float]:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.
        y : 1D np.ndarray of int, of shape [n_cases] - class labels (ground truth)
            indices correspond to instance indices in X

        Returns
        -------
        Tuple of floats, harmonic mean, accuracy and earliness scores of predict(X) vs y
        """
        self._check_is_fitted()
        X = self._preprocess_collection(X)

        return self._score(X, y)

    def get_state_info(self):
        """Return the state information generated from the last predict/update call.

        Returns
        -------
        An array containing the state info for each decision in X from update and
        predict methods. Contains classifier dependant information for future decisions
        on the data and information on when a cases decision has been made. Each row
        contains information for a case from the latest decision on its safety made in
        update/predict. Successive updates are likely to remove rows from the
        state_info, as it will only store as many rows as there are input instances to
        update/predict.
        """
        return self.state_info

    def reset_state_info(self):
        """Reset the state information used in update methods."""
        self.state_info = None

    @staticmethod
    def filter_X(X, decisions):
        """Remove True cases from X given a boolean array of decisions."""
        inv_dec = np.invert(decisions)
        return X[inv_dec]

    @staticmethod
    def filter_X_y(X, y, decisions):
        """Remove True cases from X and y given a boolean array of decisions."""
        inv_dec = np.invert(decisions)
        return X[inv_dec], y[inv_dec]

    @staticmethod
    def split_indices(indices, decisions):
        """Split a list of indices given a boolean array of decisions."""
        inv_dec = np.invert(decisions)
        return indices[inv_dec], indices[decisions]

    @staticmethod
    def split_indices_and_filter(X, indices, decisions):
        """Remove True cases and split a list of indices given an array of decisions."""
        inv_dec = np.invert(decisions)
        return X[inv_dec], indices[inv_dec], indices[decisions]

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.
        y : 1D np.array of int, of shape [n_cases] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        ...

    @abstractmethod
    def _predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        This method should update state_info with any values necessary to make future
        decisions. It is recommended that the previous time stamp used for each case
        should be stored in the state_info. The number of rows in state_info after the
        method has been called should match the number of input rows.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 1D np.array of int, of shape [n_cases] - predicted class labels
            indices correspond to instance indices in X
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        ...

    @abstractmethod
    def _update_predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Update label prediction for sequences in X at a larger series length.

        Abstract method, must be implemented.

        Uses information from previous decisions stored in state_info. This method
        should update state_info with any values necessary to make future decisions.
        It is recommended that the previous time stamp used for each case should be
        stored in the state_info. The number of rows in state_info after the method has
        been called should match the number of input rows.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 1D np.array of int, of shape [n_cases] - predicted class labels
            indices correspond to instance indices in X
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        ...

    def _predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predicts labels probabilities for sequences in X.

        This method should update state_info with any values necessary to make future
        decisions. It is recommended that the previous time stamp used for each case
        should be stored in the state_info. The number of rows in state_info after the
        method has been called should match the number of input rows.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0 if a positive decision is made. Override if
        better estimates are obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds, decisions = self._predict(X)
        for i in range(0, X.shape[0]):
            if decisions[i]:
                dists[i, self._class_dictionary[preds[i]]] = 1
            else:
                dists[i, :] = -1

        return dists, decisions

    def _update_predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Update label probabilities for sequences in X at a larger series length.

        Uses information from previous decisions stored in state_info. This method
        should update state_info with any values necessary to make future decisions.
        It is recommended that the previous time stamp used for each case should be
        stored in the state_info. The number of rows in state_info after the method has
        been called should match the number of input rows.

        Default behaviour is to call _update_predict and set the predicted class
        probability to 1, other class probabilities to 0 if a positive decision is made.
        Override if better estimates are obtainable.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : 1D bool array
            An array of booleans, containing the decision of whether a prediction is
            safe to use or not.
            i-th entry is the classifier decision that i-th instance safe to use
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds, decisions = self._update_predict(X)
        for i in range(0, X.shape[0]):
            if decisions[i]:
                dists[i, self._class_dictionary[preds[i]]] = 1
            else:
                dists[i, :] = -1

        return dists, decisions

    @abstractmethod
    def _score(self, X, y) -> tuple[float, float, float]:
        """Scores predicted labels against ground truth labels on X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : 3D np.ndarray
            Input data, any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)``
            or 2D np.array (univariate, equal length series) of shape
            ``(n_cases, n_timepoints)``
            or list of numpy arrays (any number of channels, unequal length series)
            of shape ``[n_cases]``, 2D np.array ``(n_channels, n_timepoints_i)``,
            where ``n_timepoints_i`` is length of series ``i``. other types are
            allowed and converted into one of the above.
        y : 1D np.array of int, of shape [n_cases] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        Tuple of floats, harmonic mean, accuracy and earliness scores of predict(X) vs y
        """
        ...
