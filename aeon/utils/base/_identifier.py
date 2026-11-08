"""Utility to determine type identifier of estimator, based on base class."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["get_identifier"]

from inspect import isclass

from aeon.base import BaseAeonEstimator
from aeon.utils.base._register import BASE_CLASS_REGISTER


def get_identifier(estimator):
    """Determine identifier string of an estimator.

    The identifier is the key of the most specific base class in BASE_CLASS_REGISTER
    that the estimator inherits from, i.e. "classifier" rather than
    "collection-estimator".

    Parameters
    ----------
    estimator : class or object BaseAeonEstimator
        The estimator to determine the identifier for.

    Returns
    -------
    identifier : str
         aeon identifier string from BASE_CLASS_REGISTER for the estimator.

    Raises
    ------
    TypeError
        If estimator is not a BaseAeonEstimator, or if it inherits from more than one
        unrelated base class in BASE_CLASS_REGISTER.
    """
    if isinstance(estimator, BaseAeonEstimator):
        estimator = type(estimator)
    elif not (isclass(estimator) and issubclass(estimator, BaseAeonEstimator)):
        raise TypeError(
            "Estimator must be an instance or subclass of BaseAeonEstimator."
        )

    bases = {
        key: value
        for key, value in BASE_CLASS_REGISTER.items()
        if issubclass(estimator, value)
    }
    # keep only bases which are not a parent of another matched base
    identifiers = [
        key
        for key, value in bases.items()
        if not any(
            other is not value and issubclass(other, value) for other in bases.values()
        )
    ]

    if len(identifiers) > 1:
        raise TypeError(
            f"Error, multiple identifiers could be determined for obj: {identifiers}"
        )

    return identifiers[0]
