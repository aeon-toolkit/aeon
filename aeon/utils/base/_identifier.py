"""Utility to determine type identifier of estimator, based on base class."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["get_identifier"]

from inspect import isclass

from aeon.utils.base import BASE_CLASS_REGISTER


def get_identifier(estimator):
    """Determine identifier string of an estimator.

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
        If no identifier can be determined for obj
    """
    if isclass(estimator):
        identifiers = [
            id
            for id in BASE_CLASS_REGISTER.values()
            if isinstance(estimator, id) or issubclass(estimator, id)
        ]
    else:
        identifiers = [
            id for id in BASE_CLASS_REGISTER.values() if isinstance(estimator, id)
        ]

    if len(identifiers) == 0:
        raise TypeError("Error, no identifiers could be determined for obj")

    if len(identifiers) > 1 and "estimator" in identifiers:
        identifiers.remove("estimator")
    if len(identifiers) > 1 and "series-estimator" in identifiers:
        identifiers.remove("series-estimator")
    if len(identifiers) > 1 and "collection-estimator" in identifiers:
        identifiers.remove("collection-estimator")
    if len(identifiers) > 1 and "transformer" in identifiers:
        identifiers.remove("transformer")

    if len(identifiers) > 1:
        TypeError(
            f"Error, multiple identifiers could be determined for obj: {identifiers}"
        )

    return identifiers[0]
