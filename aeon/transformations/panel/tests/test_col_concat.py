# -*- coding: utf-8 -*-
from aeon.datasets import load_basic_motions
from aeon.transformations.panel.compose import ColumnConcatenator


def test_TimeSeriesConcatenator():
    X, y = load_basic_motions(split="train")
    n_cases, n_channels, series_length = X.shape
    trans = ColumnConcatenator()
    Xt = trans.fit_transform(X)

    # check if transformed dataframe is univariate
    assert Xt.shape[1] == 1

    # check if number of time series observations are correct
    assert Xt.shape[2] == X.shape[1] * X.shape[2]

    # check specific observations
    assert X[0][-1][-3] == Xt[0][0][-3]
    assert X[0][0][3] == Xt[0, 0][3]
