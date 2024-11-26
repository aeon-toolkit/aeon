"""Test interval pipelines."""

import pytest
from sklearn.svm import SVR

from aeon.regression.interval_based import RandomIntervalRegressor
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_REGRESSION
from aeon.testing.utils.estimator_checks import _assert_predict_labels


@pytest.mark.parametrize("cls", [RandomIntervalRegressor])
def test_interval_pipeline_classifiers(cls):
    """Test the random interval regressors."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"]["train"]
    X_test, y_test = EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"]["test"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"estimator": SVR()})

    reg = cls(**params)
    reg.fit(X_train, y_train)
    prob = reg.predict(X_test)
    _assert_predict_labels(prob, X_test)
