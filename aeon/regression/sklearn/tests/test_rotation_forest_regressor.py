"""Rotation Forest test code."""

import numpy as np
from sklearn.metrics import mean_squared_error

from aeon.datasets import load_covid_3month
from aeon.regression.sklearn import RotationForestRegressor


def test_contracted_rotf():
    """Test of RotF contracting and train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")

    rotf = RotationForestRegressor(
        time_limit_in_minutes=5,
        contract_max_n_estimators=5,
        save_transformed_data=True,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    assert len(rotf.estimators_) > 0

    # test train estimate
    train_preds = rotf._get_train_preds(X_train, y_train)
    assert isinstance(train_preds, np.ndarray)
    assert train_preds.shape[0] == len(X_train)

    X_test, y_test = load_covid_3month(split="test", return_type="numpy2d")

    y_pred = rotf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    np.testing.assert_almost_equal(mse, 0.002, decimal=4)
