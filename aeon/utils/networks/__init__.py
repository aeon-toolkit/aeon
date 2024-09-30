"""Utils for tensorflow_addons."""

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"]):
    from aeon.utils.networks.weight_norm import WeightNormalization

    __all__ = ["WeightNormalization"]
