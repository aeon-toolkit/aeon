"""Utility to determine type identifier of estimator, based on base class type."""

__maintainer__ = []

from inspect import isclass

from aeon.registry._base_classes import BASE_CLASS_REGISTER


def get_identifiers(obj, force_single_identifier=True, coerce_to_list=False):
    """Determine identifier string of obj.

    Parameters
    ----------
    obj : class or object inheriting from aeon BaseObject
    force_single_identifier : bool, optional, default = True
        whether only a single identifier is returned
        if True, only the *first* identifier found will be returned
        order is determined by the order in BASE_CLASS_REGISTER
    coerce_to_list : bool, optional, default = False
        whether return should be coerced to list, even if only one identifier is
        found.

    Returns
    -------
    str, or list of str
    Identifiers from BASE_CLASS_REGISTER. aeon identifier string, if exactly one
    identifier can be determined for obj or force_single_identifier is True,
    and if coerce_to_list is False. obj has identifier if it inherits from class in
    same row of BASE_CLASS_REGISTER

    Raises
    ------
    TypeError if no identifier can be determined for obj
    """
    if isclass(obj):
        identifiers = [id[0] for id in BASE_CLASS_REGISTER if issubclass(obj, id[1])]
    else:
        identifiers = [id[0] for id in BASE_CLASS_REGISTER if isinstance(obj, id[1])]

    if len(identifiers) == 0:
        raise TypeError("Error, no identifiers could be determined for obj")

    if len(identifiers) > 1 and "object" in identifiers:
        identifiers = list(set(identifiers).difference(["object"]))

    if len(identifiers) > 1 and "estimator" in identifiers:
        identifiers = list(set(identifiers).difference(["estimator"]))

    if len(identifiers) > 1 and "series-estimator" in identifiers:
        identifiers = list(set(identifiers).difference(["series-estimator"]))
    if len(identifiers) > 1 and "collection-estimator" in identifiers:
        identifiers = list(set(identifiers).difference(["collection-estimator"]))
    if len(identifiers) > 1 and "collection-estimator" in identifiers:
        identifiers = list(set(identifiers).difference(["series-estimator"]))
    # remove transformer if collection-transformer is present
    if len(identifiers) > 1 and "collection-transformer" in identifiers:
        identifiers = list(set(identifiers).difference(["transformer"]))
    if len(identifiers) > 1 and "series-transformer" in identifiers:
        identifiers = list(set(identifiers).difference(["transformer"]))

    if force_single_identifier:
        identifiers = [identifiers[0]]

    if len(identifiers) == 1 and not coerce_to_list:
        return identifiers[0]

    return identifiers
