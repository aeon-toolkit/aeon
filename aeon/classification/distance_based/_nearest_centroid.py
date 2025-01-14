import numpy as np

from aeon.classification.base import BaseClassifier
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.distances import pairwise_distance


class NearestCentroid(BaseClassifier):
    def __init__(
        self,
        distance: str = "soft_dtw",
        average_method: str = "mean",
        distance_params: dict = None,
        average_params: dict = None,
    ):
        self.distance = distance
        self.distance_params = distance_params
        self.average_method = average_method
        self.average_params = average_params

        self.centroids_ = None

        super().__init__()

    def _fit(self, X, y):
        distance_params = self.distance_params
        average_params = self.average_params
        if distance_params is None:
            distance_params = {}
        if average_params is None:
            average_params = {}

        averaging_method = _resolve_average_callable(self.average_method)

        self.centroids_ = {}

        classes = np.unique(y)

        average_combined_params = {**average_params, **distance_params}
        for cls in classes:
            class_instances = X[y == cls]

            self.centroids_[cls] = averaging_method(
                class_instances, distance=self.distance, **average_combined_params
            )

        self.classes_ = classes

    def _predict(self, X) -> np.ndarray:
        distance_params = self.distance_params
        if distance_params is None:
            distance_params = {}
        if isinstance(self.distance, str):
            pairwise_matrix = pairwise_distance(
                X,
                np.array(list(self.centroids_.values())),
                method=self.distance,
                **distance_params,
            )
        else:
            pairwise_matrix = pairwise_distance(
                X,
                np.array(list(self.centroids_.values())),
                method=self.distance,
                **distance_params,
            )

        nearest_centroids = pairwise_matrix.argmin(axis=1)

        # Map indices to class labels
        return np.array([self.classes_[idx] for idx in nearest_centroids])


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
#                 )
#                 classifier.fit(X, y)
#                 y_pred = classifier.predict(X_test)
#                 print(
#                     f"distance: {distance}: gamma: {gamma}: score
#                     {accuracy_score(y_test, y_pred)}"
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
