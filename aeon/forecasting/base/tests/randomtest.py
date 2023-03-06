# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for random_state Parameter."""

__author__ = ["Ris-Bali"]
import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.ets import AutoETS
from aeon.forecasting.exp_smoothing import ExponentialSmoothing
from aeon.forecasting.sarimax import SARIMAX
from aeon.forecasting.structural import UnobservedComponents
from aeon.forecasting.var import VAR
from aeon.utils._testing.forecasting import make_forecasting_problem

fh = np.arange(1, 5)

y = load_airline()
y_1 = make_forecasting_problem(n_columns=3)


@pytest.mark.parametrize(
    "model",
    [AutoETS, ExponentialSmoothing, SARIMAX, UnobservedComponents, VAR],
)
def test_random_state(model):
    """Function to test random_state parameter."""
    obj = model.create_test_instance()
    if model == VAR:
        obj.fit(y=y_1, fh=fh)
        y = obj.predict()
        obj.fit(y=y_1, fh=fh)
        y1 = obj.predict()
    else:
        obj.fit(y=y, fh=fh)
        y = obj.predict()
        obj.fit(y=y, fh=fh)
        y1 = obj.predict()
    assert y == y1
