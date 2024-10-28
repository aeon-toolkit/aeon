"""Tests for BKfilter wrapper annotation estimator."""

__maintainer__ = []

import numpy as np


def test_bkilter():
    """Verify that the wrapped BKFilter estimator agrees with statsmodel.

    Expected values generated with statsmodels with this code:

    import statsmodels.api as sm
    import pandas as pd
    dta = sm.datasets.macrodata.load_pandas().data
    index = pd.Index(sm.tsa.datetools.dates_from_range("1959Q1", "2009Q3"))
    dta.set_index(index, inplace=True)
    dta.index = index
    del dta["year"]
    del dta["quarter"]
    print(type(dta[['realinv']]))
    print(dta[['realinv']].shape)
    X = dta[['realinv']].head(40).to_numpy()
    """
    X = np.array(
        [
            [
                286.89800,
                310.85900,
                289.22600,
                299.35600,
                331.72200,
                298.15200,
                296.37500,
                259.76400,
                266.40500,
                286.24600,
                310.22700,
                315.46300,
                334.27100,
                331.03900,
                336.96200,
                325.65000,
                343.72100,
                348.73000,
                360.10200,
                364.53400,
                379.52300,
                377.77800,
                386.75400,
                389.91000,
                429.14500,
                429.11900,
                444.44400,
                446.49300,
                484.24400,
                475.40800,
                470.69700,
                472.95700,
                460.00700,
                440.39300,
                453.03300,
                462.83400,
                472.90700,
                492.02600,
                476.05300,
                480.99800,
            ]
        ]
    )
    expected = np.array(
        [
            [
                7.32081,
                2.88691,
                -6.81898,
                -13.49436,
                -13.27936,
                -9.40591,
                -5.69109,
                -5.13308,
                -7.27347,
                -9.24337,
                -8.48292,
                -4.44776,
                2.40656,
                10.68433,
                19.46414,
                28.09749,
            ]
        ]
    )
    from aeon.transformations.series._bkfilter import BKFilter

    bk = BKFilter()
    X2 = bk.fit_transform(X)
    np.testing.assert_almost_equal(expected, X2, decimal=4)
