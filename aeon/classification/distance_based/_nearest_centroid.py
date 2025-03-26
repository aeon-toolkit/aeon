import numpy as np
from numba import set_num_threads

from aeon.classification.base import BaseClassifier
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.distances import pairwise_distance
from aeon.utils.validation import check_n_jobs


class NearestCentroid(BaseClassifier):
    def __init__(
        self,
        distance: str = "soft_dtw",
        average_method: str = "mean",
        distance_params: dict = None,
        average_params: dict = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.average_method = average_method
        self.average_params = average_params
        self.verbose = verbose
        self.n_jobs = n_jobs

        self._distance_params = None
        self._average_params = None
        self._averaging_method = None

        self.centroids_ = None

        super().__init__()

    def _fit(self, X, y):
        self._check_params()
        self.centroids_ = {}

        classes = np.unique(y)

        for cls in classes:
            class_instances = X[y == cls]

            self.centroids_[cls] = self._averaging_method(
                class_instances, **self._average_params
            )

        self.classes_ = classes

    def _predict(self, X) -> np.ndarray:
        pairwise_matrix = pairwise_distance(
            X,
            np.array(list(self.centroids_.values())),
            method=self.distance,
            **self._distance_params,
        )
        nearest_centroids = pairwise_matrix.argmin(axis=1)

        # Map indices to class labels
        return np.array([self.classes_[idx] for idx in nearest_centroids])

    def _check_params(self):
        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        if self.distance_params is None:
            self._distance_params = {}
        else:
            self._distance_params = self.distance_params
        if self.average_params is None:
            self._average_params = {}
        else:
            self._average_params = self.average_params

        self._average_params = {
            "n_jobs": self._n_jobs,
            "distance": self.distance,
            "verbose": self.verbose,
            **self._average_params,
            **self._distance_params,
        }

        self._averaging_method = _resolve_average_callable(self.average_method)


# if __name__ == "__main__":
#     from sklearn.metrics import accuracy_score
#
#     from aeon.datasets import load_gunpoint
#
#     X, y = load_gunpoint(split="train")
#     X_test, y_test = load_gunpoint(split="test")
#
#     experiments = [
#         ["euclidean", "mean"],
#         ["dtw", "ba"],
#         ["msm", "ba"],
#         ["twe", "ba"],
#         ["soft_dtw", "soft_ba"],
#         ["soft_msm", "soft_ba"],
#         ["soft_twe", "soft_ba"],
#     ]
#
#     for distance, average in experiments:
#
#         if "soft" in distance:
#             for gamma in [0.001, 0.01, 0.1, 1.0]:
#                 classifier = NearestCentroid(
#                     distance=distance,
#                     average_method=average,
#                     distance_params={"gamma": gamma},
#                     n_jobs=-1
#                 )
#                 classifier.fit(X, y)
#                 y_pred = classifier.predict(X_test)
#                 print(
#                     f"distance: {distance}: gamma: {gamma}: score "
#                     f"{accuracy_score(y_test, y_pred)}"
#                 )
#         else:
#             classifier = NearestCentroid(distance=distance, average_method=average)
#             classifier.fit(X, y)
#             y_pred = classifier.predict(X_test)
#             print(f"distance: {distance}: score {accuracy_score(y_test, y_pred)}")
# classifier = NearestCentroid(distance=distance, average_method=average)
# classifier.fit(X, y)
# y_pred = classifier.predict(X_test)
# print(f"distance: {distance}: score {accuracy_score(y_test, y_pred)}")
