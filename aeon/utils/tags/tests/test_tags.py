"""Tests for estimator tags."""

from aeon.utils.base import BASE_CLASS_REGISTER
from aeon.utils.tags import ESTIMATOR_TAGS


def test_estimator_tags_dict():
    """Test the estimator tags dictionary follows the correct format."""
    # check type
    assert isinstance(ESTIMATOR_TAGS, dict)
    assert len(ESTIMATOR_TAGS) > 0

    # check dict contents
    for key, value in ESTIMATOR_TAGS.items():
        # name and a dict containing tag information
        assert isinstance(key, str)
        assert isinstance(value, dict)
        assert "class" in value
        assert "type" in value
        assert "description" in value

        # valid estimator type
        assert isinstance(value["class"], (str, list))
        cls = value["class"] if isinstance(value["class"], list) else [value["class"]]
        for v in cls:
            assert v in BASE_CLASS_REGISTER.keys()

        # valid tag type
        assert isinstance(value["type"], (str, list, tuple))
        if isinstance(value["type"], list):
            for i in value["type"]:
                assert isinstance(i, (str, tuple)) or i is None

                if isinstance(value["type"], tuple):
                    assert len(value["type"]) == 2
                    assert value["type"][0] in ["str", "list"]
                    assert isinstance(value["type"][1], list)
                    for n in value["type"][1]:
                        assert isinstance(n, str)
        elif isinstance(value["type"], tuple):
            assert len(value["type"]) == 2
            assert value["type"][0] in ["str", "list"]
            assert isinstance(value["type"][1], list)
            for n in value["type"][1]:
                assert isinstance(n, str)

        # valid description
        assert isinstance(value["description"], str)
