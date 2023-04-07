# -*- coding: utf-8 -*-
"""Base classes for adapting other forecasters to aeon framework."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "_ProphetAdapter",
    "_StatsModelsAdapter",
    "_TbatsAdapter",
    "_PmdArimaAdapter",
    "_StatsForecastAdapter",
]

from aeon.forecasting.base.adapters._fbprophet import _ProphetAdapter
from aeon.forecasting.base.adapters._pmdarima import _PmdArimaAdapter
from aeon.forecasting.base.adapters._statsforecast import _StatsForecastAdapter
from aeon.forecasting.base.adapters._statsmodels import _StatsModelsAdapter
from aeon.forecasting.base.adapters._tbats import _TbatsAdapter
