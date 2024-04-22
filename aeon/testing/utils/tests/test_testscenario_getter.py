"""Test suite for retrieving scenarios."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.registry import BASE_CLASS_IDENTIFIER_LIST, BASE_CLASS_LIST
from aeon.testing.utils.scenarios import TestScenario
from aeon.testing.utils.scenarios_getter import retrieve_scenarios


@pytest.mark.parametrize("estimator_class", BASE_CLASS_LIST)
def test_get_scenarios_for_class(estimator_class):
    """Test retrieval of scenarios by class."""
    scenarios = retrieve_scenarios(obj=estimator_class)

    assert isinstance(scenarios, list), "return of retrieve_scenarios is not a list"
    assert np.all(
        isinstance(x, TestScenario) for x in scenarios
    ), "return of retrieve_scenarios is not a list of scenarios"


@pytest.mark.parametrize("type_string", BASE_CLASS_IDENTIFIER_LIST)
def test_get_scenarios_for_string(type_string):
    """Test retrieval of scenarios by string."""
    scenarios = retrieve_scenarios(obj=type_string)

    assert isinstance(scenarios, list), "return of retrieve_scenarios is not a list"
    assert np.all(
        isinstance(x, TestScenario) for x in scenarios
    ), "return of retrieve_scenarios is not a list of scenarios"


def test_get_scenarios_errors():
    """Test that errors are raised for bad input args."""
    with pytest.raises(TypeError):
        retrieve_scenarios()

    with pytest.raises(TypeError):
        retrieve_scenarios(obj=1)
