import pytest

from aeon.registry import BASE_CLASS_IDENTIFIER_LIST
from aeon.utils._lookup import _check_estimator_types
from aeon.utils.base._identifier import get_identifiers


@pytest.mark.parametrize("estimator_id", BASE_CLASS_IDENTIFIER_LIST)
def test_type_inference(estimator_id):
    """Check that identifier inverts _check_estimator_types."""
    base_class = _check_estimator_types(estimator_id)[0]
    inferred_type = get_identifiers(base_class)

    assert (
        inferred_type == estimator_id
    ), "one of types _check_estimator_types is incorrect, these should be inverses"
