"""Wrapper of KNeighborsTimeSeriesClassifier named Single_Class_KNN.

It wraps the fit setup to ensure `_fit` is executed even when the dataset
contains only a single class.
"""

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

__all__ = ["Single_Class_KNN"]


class Single_Class_KNN(KNeighborsTimeSeriesClassifier):
    """
    KNN classifier for time series data, adapted to work with SMOTE.

    This class is a wrapper around the original KNeighborsTimeSeriesClassifier
    to ensure compatibility with the Signal class.
    """

    def _fit_setup(self, X, y):
        # KNN can support if all labels are the same so always return False for single
        # class problem so the fit will always run
        X, y, _ = super()._fit_setup(X, y)
        return X, y, False
