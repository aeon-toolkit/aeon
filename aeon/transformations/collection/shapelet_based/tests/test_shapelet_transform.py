# -*- coding: utf-8 -*-
"""ShapeletTransform test code."""
import numpy as np
from numpy import testing

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform


def test_st_on_unit_test():
    """Test of ShapeletTransform on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, random_state=0
    )
    st.fit(X_train[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train[indices])
    testing.assert_array_almost_equal(
        data, shapelet_transform_unit_test_data, decimal=4
    )


def test_st_on_basic_motions():
    """Test of ShapeletTransform on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, random_state=0
    )
    st.fit(X_train[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train[indices])
    testing.assert_array_almost_equal(
        data, shapelet_transform_basic_motions_data, decimal=4
    )


shapelet_transform_unit_test_data = np.array(
    [
        [0.0845, 0.1536, 0.1812],
        [0.1006, 0.1373, 0.1423],
        [0.1557, 0.2457, 0.2811],
        [0.1437, 0.1117, 0.0832],
        [0.059, 0.117, 0.1461],
    ]
)
shapelet_transform_basic_motions_data = np.array(
    [
        [
            1.09081752,
            0.97608253,
            1.27111764,
            2.15177841,
            1.28195108,
            1.62195992,
            1.33683766,
        ],
        [
            1.16752113,
            1.04783354,
            1.08418559,
            0.98715225,
            0.72011113,
            1.03197761,
            1.10632284,
        ],
        [
            1.63217996,
            1.99463377,
            2.080681,
            1.95568228,
            2.33019465,
            1.33627307,
            0.39141083,
        ],
        [
            1.13716252,
            1.00537502,
            0.67820667,
            0.41299391,
            1.0092179,
            1.0389656,
            1.04563195,
        ],
        [
            0.33453983,
            0.275241,
            1.54179706,
            2.16604241,
            1.20134733,
            1.6179654,
            1.55278425,
        ],
    ]
)
