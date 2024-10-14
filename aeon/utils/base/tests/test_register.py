"""Test the base class registers follow the correct format."""

from aeon.base import BaseAeonEstimator
from aeon.utils.base._register import BASE_CLASS_REGISTER, VALID_ESTIMATOR_BASES


def test_base_class_register():
    """Test the base class registers follow the correct format."""
    assert isinstance(BASE_CLASS_REGISTER, dict)
    assert len(BASE_CLASS_REGISTER) > 0
    assert all(isinstance(k, str) for k in BASE_CLASS_REGISTER.keys())
    assert all(
        issubclass(v, BaseAeonEstimator) or isinstance(v, BaseAeonEstimator)
        for v in BASE_CLASS_REGISTER.values()
    )

    assert len(VALID_ESTIMATOR_BASES) < len(BASE_CLASS_REGISTER)
    assert BaseAeonEstimator not in VALID_ESTIMATOR_BASES.values()
