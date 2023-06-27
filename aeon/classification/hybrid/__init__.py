# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Hybrid time series classifiers."""

__all__ = [
    "HIVECOTEV1",
    "HIVECOTEV2",
]

from aeon.classification.hybrid._hivecote_v1 import HIVECOTEV1
from aeon.classification.hybrid._hivecote_v2 import HIVECOTEV2
