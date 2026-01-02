"""Utilities for class __repr__ presentation."""

import inspect

from aeon.testing.utils.deep_equals import deep_equals

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "get_unchanged_and_required_params_as_str",
]


def get_unchanged_and_required_params_as_str(obj):
    """
    Get object parameters as a comma delimited string.

    Collects the parameters of an object that are either required
    (no default) or different from the __init__ default value. Returns
    the parameter names and values as a comma delimited string.

    Parameters
    ----------
    obj : object
        The object to inspect.

    Returns
    -------
    str
        A string representation of the objects parameters and values.
    """
    cls = obj.__class__
    signature = inspect.signature(cls.__init__)

    params = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue

        has_default = param.default is not inspect.Parameter.empty
        current_val = getattr(obj, name, None)

        if not has_default:
            # No default = always include
            params[name] = current_val
        else:
            # Default exists = include if unchanged
            if not deep_equals(current_val, param.default):
                params[name] = current_val

    if len(params) == 0:
        return ""

    param_str = []
    for k, v in params.items():
        if isinstance(v, str):
            param_str.append(f"{k}='{v}'")
        else:
            param_str.append(f"{k}={v}")

    return ", ".join(param_str)
