#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Import all time series regression functionality available in sktime."""

__author__ = ["mloning"]
__all__ = [
    "np",
    "pd",
    "TimeSeriesForestRegressor",
]

import numpy as np
import pandas as pd

from aeon.regression.interval_based import TimeSeriesForestRegressor
