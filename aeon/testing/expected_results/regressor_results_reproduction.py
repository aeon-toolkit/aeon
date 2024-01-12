"""Functions for generating stored unit test results for regressors."""

import numpy as np
from sklearn.utils._testing import set_random_state

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
    indices_train = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
    indices_test = np.random.RandomState(0).choice(len(y_test), 10, replace=False)

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


def _print_results_for_regressor(regressor_name, dataset_name):
    if regressor_name == "FreshPRINCERegressor":
        regressor = FreshPRINCERegressor.create_test_instance(
            parameter_set="results_comparison"
        )
    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")

    if dataset_name == "Covid3Month":
        data_function = _reproduce_regression_covid_3month
    elif dataset_name == "CardanoSentiment":
        data_function = _reproduce_regression_cardano_sentiment
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    set_random_state(regressor, 0)

    _print_array(
        f"{regressor_name} - {dataset_name}",
        data_function(regressor),
    )


if __name__ == "__main__":
    # change as required when adding new classifiers, datasets or updating results
    _print_results_for_regressor("FreshPRINCERegressor", "Covid3Month")
