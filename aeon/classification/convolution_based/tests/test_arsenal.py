# -*- coding: utf-8 -*-
"""Arsenal test code."""
from aeon.classification.convolution_based import Arsenal
from aeon.utils._testing.collection import make_2d_test_data, make_3d_test_data


def test_contracted_arsenal():
    """Test of contracted Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = make_2d_test_data()
    # train contracted Arsenal
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=5,
        num_kernels=20,
        random_state=0,
    )
    arsenal.fit(X_train, y_train)
    assert len(arsenal.estimators_) > 1
    X_train, y_train = make_3d_test_data()
    arsenal.fit(X_train, y_train)
    assert len(arsenal.estimators_) > 1
