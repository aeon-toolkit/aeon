"""Testing of registry lookup functionality."""

__author__ = ["fkiraly", "MatthewMiddlehurst"]

import pytest

from aeon.base import BaseObject
from aeon.registry import all_estimators, all_tags, get_identifiers
from aeon.registry._base_classes import BASE_CLASS_IDENTIFIER_LIST, BASE_CLASS_LOOKUP
from aeon.registry._lookup import _check_estimator_types
from aeon.transformations.base import BaseTransformer

VALID_IDENTIFIERS_SET = set(BASE_CLASS_IDENTIFIER_LIST + ["estimator"])

CLASSES_WITHOUT_TAGS = [
    "series-annotator",
    "object",
    "splitter",
    "network",
    "collection-transformer",
    "collection-estimator",
    "similarity-search",
]

# shorthands for easy reading
b = BASE_CLASS_IDENTIFIER_LIST
n = len(b)

# selected examples of "search for two types at once to avoid quadratic scaling"
double_estimators = [[b[i], b[(i + 3) % n]] for i in range(n)]
# fixtures search by individual identifiers, "None", and some pairs
estimator_fixture = [None] + BASE_CLASS_IDENTIFIER_LIST + double_estimators


def _to_list(obj):
    """Put obj in list if it is not a list."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def _get_type_tuple(estimator_identifiers):
    """Convert string(s) into tuple of classes for isinstance check.

    Parameters
    ----------
    estimator_identifiers: None, string, or list of string

    Returns
    -------
    estimator_classes : tuple of aeon base classes,
        corresponding to strings in estimator_identifiers
    """
    if estimator_identifiers is not None:
        estimator_classes = tuple(
            BASE_CLASS_LOOKUP[id] for id in _to_list(estimator_identifiers)
        )
    else:
        estimator_classes = (BaseObject,)

    return estimator_classes


@pytest.mark.parametrize("return_names", [True, False])
@pytest.mark.parametrize("estimator_id", estimator_fixture)
def test_all_estimators_by_identifier(estimator_id, return_names):
    """Check that all_estimators return argument has correct type."""
    estimators = all_estimators(
        estimator_types=estimator_id,
        return_names=return_names,
    )

    estimator_classes = _get_type_tuple(estimator_id)

    assert isinstance(estimators, list)
    # there should be at least one estimator returned
    assert len(estimators) > 0

    # checks return type specification (see docstring)
    if return_names:
        for estimator in estimators:
            assert isinstance(estimator, tuple) and len(estimator) == 2
            assert isinstance(estimator[0], str)
            assert issubclass(estimator[1], estimator_classes)
            assert estimator[0] == estimator[1].__name__
    else:
        for estimator in estimators:
            assert issubclass(estimator, estimator_classes)


@pytest.mark.parametrize("estimator_id", estimator_fixture)
def test_all_tags(estimator_id):
    """Check that all_tags return argument has correct type."""
    tags = all_tags(estimator_types=estimator_id)
    assert isinstance(tags, list)

    # there should be at least one tag returned
    # exception: types which we know don't have tags associated
    est_list = estimator_id if isinstance(estimator_id, list) else [estimator_id]
    if not set(est_list).issubset(CLASSES_WITHOUT_TAGS):
        assert len(tags) > 0

    # checks return type specification (see docstring)
    for tag in tags:
        assert isinstance(tag, tuple)
        assert isinstance(tag[0], str)
        assert VALID_IDENTIFIERS_SET.issuperset(_to_list(tag[1]))
        assert isinstance(tag[2], (str, tuple))
        if isinstance(tag[2], tuple):
            assert len(tag[2]) == 2
            assert isinstance(tag[2][0], str)
            assert isinstance(tag[2][1], (str, list))
        assert isinstance(tag[3], str)


@pytest.mark.parametrize("return_names", [True, False])
def test_all_estimators_return_names(return_names):
    """Test return_names argument in all_estimators."""
    estimators = all_estimators(return_names=return_names)
    assert isinstance(estimators, list)
    assert len(estimators) > 0

    if return_names:
        assert all([isinstance(estimator, tuple) for estimator in estimators])
        names, estimators = list(zip(*estimators))
        assert all([isinstance(name, str) for name in names])
        assert all(
            [name == estimator.__name__ for name, estimator in zip(names, estimators)]
        )

    assert all([isinstance(estimator, type) for estimator in estimators])


def test_all_estimators_exclude_type():
    """Test exclude_estimator_types argument in all_estimators."""
    estimators = all_estimators(
        return_names=True, exclude_estimator_types="transformer"
    )
    assert isinstance(estimators, list)
    assert len(estimators) > 0
    names, estimators = list(zip(*estimators))

    for estimator in estimators:
        assert not isinstance(estimator, BaseTransformer)


# arbitrary list for exclude_estimators argument test
EXCLUDE_ESTIMATORS = [
    "ElasticEnsemble",
    "NaiveForecaster",
]


@pytest.mark.parametrize("exclude_estimators", ["NaiveForecaster", EXCLUDE_ESTIMATORS])
def test_all_estimators_exclude_estimators(exclude_estimators):
    """Test exclued_estimators argument in all_estimators."""
    estimators = all_estimators(
        return_names=True, exclude_estimators=exclude_estimators
    )
    assert isinstance(estimators, list)
    assert len(estimators) > 0
    names, estimators = list(zip(*estimators))

    if not isinstance(exclude_estimators, list):
        exclude_estimators = [exclude_estimators]
    for estimator in exclude_estimators:
        assert estimator not in names


def _get_tag_fixture():
    """Generate a simple list of test cases for optional return_tags."""
    # just picked a few valid tags to try out as valid str return_tags args:
    test_str_as_arg = [
        "X-y-must-have-same-index",
        "capability:pred_var",
        "skip-inverse-transform",
    ]

    # we can also make them into a list to test list of str as a valid arg:
    test_list_as_arg = [test_str_as_arg]
    # Note - I don't include None explicitly as a test case - tested elsewhere
    return test_str_as_arg + test_list_as_arg


# test that all_estimators returns as expected if given correct return_tags:
@pytest.mark.parametrize("return_tags", _get_tag_fixture())
@pytest.mark.parametrize("return_names", [True, False])
def test_all_estimators_return_tags(return_tags, return_names):
    """Test ability to return estimator value of passed tags."""
    estimators = all_estimators(
        return_tags=return_tags,
        return_names=return_names,
    )
    # Helps us keep track of estimator index within the tuple:
    ESTIMATOR_INDEX = 1 if return_names else 0
    TAG_START_INDEX = ESTIMATOR_INDEX + 1

    assert isinstance(estimators[0], tuple)
    # check length of tuple is what we expect:
    if isinstance(return_tags, str):
        assert len(estimators[0]) == TAG_START_INDEX + 1
    else:
        assert len(estimators[0]) == len(return_tags) + TAG_START_INDEX

    # check that for each estimator the value for that tag is correct:
    for est_tuple in estimators:
        est = est_tuple[ESTIMATOR_INDEX]
        if isinstance(return_tags, str):
            assert est.get_class_tag(return_tags) == est_tuple[TAG_START_INDEX]
        else:
            for tag_index, tag in enumerate(return_tags):
                assert est.get_class_tag(tag) == est_tuple[TAG_START_INDEX + tag_index]


def _get_bad_return_tags():
    """Get return_tags arguments that should throw an exception."""
    # case not a str or a list:
    is_int = [12]
    # case is a list, but not all elements are str:
    is_not_all_str = [["this", "is", "a", "test", 12, "!"]]

    return is_int + is_not_all_str


# test that all_estimators breaks as expected if given bad return_tags:
@pytest.mark.parametrize("return_tags", _get_bad_return_tags())
def test_all_estimators_return_tags_bad_arg(return_tags):
    """Test ability to catch bad arguments of return_tags."""
    with pytest.raises(TypeError):
        _ = all_estimators(return_tags=return_tags)


@pytest.mark.parametrize("estimator_id", BASE_CLASS_IDENTIFIER_LIST)
def test_type_inference(estimator_id):
    """Check that identifier inverts _check_estimator_types."""
    base_class = _check_estimator_types(estimator_id)[0]
    inferred_type = get_identifiers(base_class)

    assert (
        inferred_type == estimator_id
    ), "one of types _check_estimator_types is incorrect, these should be inverses"


def test_list_tag_lookup():
    """Check that all estimators can handle tags lists rather than single strings.

    DummyClassifier has two internal datatypes, "numpy3D" and "np-list". This test
    checks that DummyClassifier is returned with either of these argument.s
    """
    matches = all_estimators(
        estimator_types="classifier", filter_tags={"X_inner_type": "np-list"}
    )
    names = [t[0] for t in matches]
    assert "DummyClassifier" in names
    matches = all_estimators(
        estimator_types="classifier", filter_tags={"X_inner_type": "numpy3D"}
    )
    names = [t[0] for t in matches]
    assert "DummyClassifier" in names
