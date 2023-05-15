# -*- coding: utf-8 -*-
import numpy as np

from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.regression.feature_based import FreshPRINCERegressor


def _reproduce_regression_covid_3month(estimator):
    X_train, y_train = load_covid_3month(split="train")
    X_test, y_test = load_covid_3month(split="test")
    indices_train = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
    indices_test = np.random.RandomState(0).choice(len(y_test), 10, replace=False)

    estimator.fit(X_train[indices_train], y_train[indices_train])
    return estimator.predict(X_test[indices_test])


def _reproduce_regression_cardano_sentiment(estimator):
    X_train, y_train = load_cardano_sentiment(split="train")
    X_test, y_test = load_cardano_sentiment(split="test")

    indices_train = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
    indices_test = np.random.RandomState(4).choice(len(y_test), 10, replace=False)

    estimator.fit(X_train[indices_train], y_train[indices_train])
    return estimator.predict(X_test[indices_test])


# flake8: noqa: T001
def _print_array(test_name, array):
    print(test_name)
    print("[")
    for value in array:
        print(str(round(value, 4)), end="")
        print(",")
    print("]")


if __name__ == "__main__":
    _print_array(
        "FreshPRINCERegressor - Covid3Month",
        _reproduce_regression_covid_3month(
            FreshPRINCERegressor(
                default_fc_parameters="minimal",
                n_estimators=10,
                random_state=0,
            )
        ),
    )
    _print_array(
        "FreshPRINCERegressor - CardanoSentiment",
        _reproduce_regression_cardano_sentiment(
            FreshPRINCERegressor(
                default_fc_parameters="minimal",
                n_estimators=10,
                random_state=0,
            )
        ),
    )
