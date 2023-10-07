"""Wrappers to control update/stream learning in continuous forecasting."""

__all__ = [
    "DontUpdate",
    "UpdateEvery",
    "UpdateRefitsEvery",
]

from aeon.forecasting.stream._update import DontUpdate, UpdateEvery, UpdateRefitsEvery
