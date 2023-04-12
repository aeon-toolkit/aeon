#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

"""Base classes for defining estimators and other objects in aeon."""

__author__ = ["mloning", "RNKuhns", "fkiraly"]
__all__ = [
    "BaseObject",
    "BaseEstimator",
    "_HeterogenousMetaEstimator",
    "load",
]

from aeon.base._base import BaseEstimator, BaseObject
from aeon.base._meta import _HeterogenousMetaEstimator
from aeon.base._serialize import load
