"""Testing of estimator lookup functionality."""

import pytest
from sklearn.base import BaseEstimator

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.base import BaseAeonEstimator
from aeon.classification import BaseClassifier, DummyClassifier
from aeon.clustering import BaseClusterer
from aeon.transformations.base import BaseTransformer
from aeon.utils.base import BASE_CLASS_REGISTER
from aeon.utils.discovery import all_estimators


def test_all_estimators():
    """Test return_names argument in all_estimators."""
    estimators = all_estimators()

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert estimator[0] == estimator[1].__name__
        assert "Base" not in estimator[0]
        assert issubclass(estimator[1], BaseEstimator)

    estimators2 = all_estimators(return_names=False)

    assert [cls for _, cls in estimators] == estimators2

    estimators3 = all_estimators(include_sklearn=False)

    assert isinstance(estimators3, list)
    assert 0 < len(estimators3) < len(estimators)

    for estimator in estimators3:
        assert issubclass(estimator[1], BaseAeonEstimator)


@pytest.mark.parametrize("item", BASE_CLASS_REGISTER.items())
def test_all_estimators_by_type(item):
    """Check that type_filter argument returns the correct type."""
    estimators = all_estimators(type_filter=item[0])

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert issubclass(estimator[1], item[1])

    estimators2 = all_estimators(type_filter=item[1])

    assert estimators == estimators2


@pytest.mark.parametrize(
    "input",
    [
        [BaseTransformer, BaseClassifier],
        [BaseClassifier, "segmenter"],
        [BaseClassifier, BaseAnomalyDetector, BaseClusterer],
    ],
)
def test_all_estimators_by_multiple_types(input):
    """Check that type_filter argument returns the correct type with multiple inputs."""
    estimators = all_estimators(type_filter=input)

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    types = tuple([i if isinstance(i, type) else BASE_CLASS_REGISTER[i] for i in input])
    for estimator in estimators:
        assert issubclass(estimator[1], types)


def test_all_estimators_exclude_type():
    """Test exclude_types argument in all_estimators."""
    estimators = all_estimators(exclude_types="transformer")

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert not isinstance(estimator[1], BaseTransformer)

    estimators2 = all_estimators(exclude_types=["transformer", "classifier"])

    assert isinstance(estimators2, list)
    assert len(estimators2) > 0 and len(estimators2) < len(estimators)

    for estimator in estimators:
        assert not isinstance(estimator[1], (BaseTransformer, BaseClassifier))


@pytest.mark.parametrize(
    "tags",
    [
        {"capability:multivariate": False},
        {"algorithm_type": "interval"},
        {"algorithm_type": "shapelet", "capability:multivariate": True},
    ],
)
def test_all_estimators_by_tag(tags):
    """Test ability to return estimator value of passed tags."""
    estimators = all_estimators(tag_filter=tags)

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert issubclass(estimator[1], BaseAeonEstimator)
        assert all(
            [estimator[1].get_class_tag(tag) == value for tag, value in tags.items()]
        )


def test_all_estimators_exclude_tags():
    """Test exclude_tags argument in all_estimators."""
    estimators = all_estimators(tag_filter={"algorithm_type": "interval"})

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert estimator[1].get_class_tag("algorithm_type") == "interval"


def test_all_estimators_filter_type_and_tag():
    """Test that type_filter and tag_filter can be used together."""
    estimators = all_estimators(
        type_filter="classifier", tag_filter={"capability:multivariate": True}
    )

    assert isinstance(estimators, list)
    assert len(estimators) > 0

    for estimator in estimators:
        assert isinstance(estimator, tuple) and len(estimator) == 2
        assert isinstance(estimator[0], str)
        assert issubclass(estimator[1], BaseClassifier)
        assert estimator[1].get_class_tag("capability:multivariate") is True

    # impossible combination
    estimators2 = all_estimators(
        type_filter="segmenter", tag_filter={"capability:unequal_length": True}
    )
    assert len(estimators2) == 0


def test_all_estimators_list_tag_lookup():
    """Check that all estimators can handle tags lists rather than single strings.

    DummyClassifier has two internal datatypes, "numpy3D" and "np-list". This test
    checks that DummyClassifier is returned with either of these arguments.
    """
    estimators = all_estimators(
        tag_filter={"X_inner_type": "np-list"},
        return_names=False,
    )
    assert DummyClassifier in estimators

    estimators2 = all_estimators(
        tag_filter={"X_inner_type": "numpy3D"},
        return_names=False,
    )
    assert DummyClassifier in estimators2

    estimators3 = all_estimators(
        tag_filter={"X_inner_type": ["np-list", "numpy3D"]},
        return_names=False,
    )
    assert DummyClassifier in estimators3

    assert len(estimators3) > len(estimators) and len(estimators3) >= len(estimators2)

    estimators4 = all_estimators(
        tag_filter={"X_inner_type": "numpy2D"},
        return_names=False,
    )
    assert DummyClassifier not in estimators4

    assert len(estimators4) > 0
