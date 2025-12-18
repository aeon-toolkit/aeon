"""Tests for the capability:predict_proba tag."""

from aeon.classification.compose import ClassifierPipeline
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.transformations.collection.unequal_length import Truncator


def test_predict_proba_tag_knn():
    """Test that KNN correctly sets the tag based on n_neighbors."""
    knn1 = KNeighborsTimeSeriesClassifier(n_neighbors=1)
    assert not knn1.get_tag("capability:predict_proba")

    knn5 = KNeighborsTimeSeriesClassifier(n_neighbors=5)
    assert knn5.get_tag("capability:predict_proba")


def test_predict_proba_tag_pipeline():
    """Test that ClassifierPipeline correctly inherits the tag."""
    transformer = Truncator(truncated_length=5)

    pipe1 = ClassifierPipeline(
        transformers=[transformer],
        classifier=KNeighborsTimeSeriesClassifier(n_neighbors=1),
    )
    assert not pipe1.get_tag("capability:predict_proba")

    pipe5 = ClassifierPipeline(
        transformers=[transformer],
        classifier=KNeighborsTimeSeriesClassifier(n_neighbors=5),
    )
    assert pipe5.get_tag("capability:predict_proba")
