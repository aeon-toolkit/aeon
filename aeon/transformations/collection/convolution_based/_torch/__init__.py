"""Convolutions based on PyTorch implementations."""

__all__ = ["_HydraInternal"]

from aeon.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("torch")

from aeon.transformations.collection.convolution_based._torch.hydra_internal import (  # noqa
    _HydraInternal,
)
