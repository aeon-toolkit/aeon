# -*- coding: utf-8 -*-
"""Wrappers to control update/stream learning in continuous forecasting."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "DontUpdate",
    "UpdateEvery",
    "UpdateRefitsEvery",
]

from aeon.forecasting.stream._update import DontUpdate, UpdateEvery, UpdateRefitsEvery
