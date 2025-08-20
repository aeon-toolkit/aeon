"""Write expected results for estimators to file for usage in tests."""

from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.base import ComposableEstimatorMixin
from aeon.base._base import _clone_estimator
from aeon.datasets import (
    load_basic_motions,
    load_cardano_sentiment,
    load_covid_3month,
    load_unit_test,
)
from aeon.utils.discovery import all_estimators

# Select an estimator type string from expected_results_types if you want to generate
# results for a specific type of estimator.
# If None, results for all types will be generated.
selected_estimator_type = None

# Estimator types to generate expected results for.
expected_results_types = ["classifier", "early_classifier", "regressor"]

# Do not produce expected results for these estimators.
# Estimators with the "non-deterministic" tag and estimators inheriting
# ComposableEstimatorMixin are excluded by default.
excluded_estimators = [
    # can't handle ordinal currently
    "OrdinalTDE",
    "IndividualOrdinalTDE",
    # wrappers
    "MrSEQLClassifier",
    "MrSQMClassifier",
    "LearningShapeletClassifier",
    # Unknown failure, needs investigation
    "SignatureClassifier",
    "TDMVDCClassifier",
    "RSASTClassifier",
    "FreshPRINCEClassifier",
    "DrCIFRegressor",
    "FreshPRINCERegressor",
]

rng = check_random_state(42)

X_ut_train, y_ut_train = load_unit_test(split="train")
ut_train_indices = rng.choice(len(y_ut_train), 15, replace=False)
X_ut_train = X_ut_train[ut_train_indices]
y_ut_train = y_ut_train[ut_train_indices]
X_ut_test, y_ut_test = load_unit_test(split="test")
ut_test_indices = rng.choice(len(y_ut_test), 10, replace=False)
X_ut_test = X_ut_test[ut_test_indices]

X_bm_train, y_bm_train = load_basic_motions(split="train")
bm_train_indices = rng.choice(len(y_bm_train), 15, replace=False)
X_bm_train = X_bm_train[bm_train_indices]
y_bm_train = y_bm_train[bm_train_indices]
X_bm_test, y_bm_test = load_basic_motions(split="test")
bm_test_indices = rng.choice(len(y_bm_test), 10, replace=False)
X_bm_test = X_bm_test[bm_test_indices]

X_c3m_train, y_c3m_train = load_covid_3month(split="train")
c3m_train_indices = rng.choice(len(y_c3m_train), 15, replace=False)
X_c3m_train = X_c3m_train[c3m_train_indices]
y_c3m_train = y_c3m_train[c3m_train_indices]
X_c3m_test, y_c3m_test = load_covid_3month(split="test")
c3m_test_indices = rng.choice(len(y_c3m_test), 10, replace=False)
X_c3m_test = X_c3m_test[c3m_test_indices]

X_cs_train, y_cs_train = load_cardano_sentiment(split="train")
cs_train_indices = rng.choice(len(y_cs_train), 15, replace=False)
X_cs_train = X_cs_train[cs_train_indices]
y_cs_train = y_cs_train[cs_train_indices]
X_cs_test, y_cs_test = load_cardano_sentiment(split="test")
cs_test_indices = rng.choice(len(y_cs_test), 10, replace=False)
X_cs_test = X_cs_test[cs_test_indices]

univariate_datasets = {
    "classifier": ((X_ut_train, y_ut_train), X_ut_test),
    "early_classifier": ((X_ut_train, y_ut_train), X_ut_test),
    "regressor": ((X_c3m_train, y_c3m_train), X_c3m_test),
    "collection-transformer": ((X_ut_train, y_ut_train), X_ut_test),
}

multivariate_datasets = {
    "classifier": ((X_bm_train, y_bm_train), X_bm_test),
    "early_classifier": ((X_bm_train, y_bm_train), X_bm_test),
    "regressor": ((X_cs_train, y_cs_train), X_cs_test),
    "collection-transformer": ((X_bm_train, y_bm_train), X_bm_test),
}


def get_expected_results(estimator, estimator_type, test_X):
    if estimator_type in ["classifier", "early_classifier"]:
        return estimator.predict_proba(test_X)
    elif estimator_type == "regressor":
        return estimator.predict(test_X)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def get_results_string(results, estimator_type):
    """Format results for printing."""
    if estimator_type == "classifier":
        s = "[\n"
        for sub_array in results:
            s += "        ["
            for i, value in enumerate(sub_array):
                s += str(round(value, 4))
                if i < len(sub_array) - 1:
                    s += ", "
            s += "],\n"
        s += "    ],\n"
    elif estimator_type == "early_classifier":
        s = "[\n"
        for sub_array in results[0]:
            s += "        ["
            for i, value in enumerate(sub_array):
                s += str(round(value, 4))
                if i < len(sub_array) - 1:
                    s += ", "
            s += "],\n"
        s += "    ],\n"
    elif estimator_type == "regressor":
        s = "[\n"
        for value in results:
            s += "        "
            s += str(round(value, 4))
            s += ",\n"
        s += "    ],\n"
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")
    return s


if __name__ == "__main__":
    for estimator_type in expected_results_types:
        if (
            selected_estimator_type is not None
            and estimator_type != selected_estimator_type
        ):
            continue

        estimators = all_estimators(type_filter=estimator_type)

        write_path = f"expected_{estimator_type.replace('-', '_')}_results.py"
        uv_results = {}
        mv_results = {}

        for estimator_name, estimator_class in estimators:
            if estimator_name in excluded_estimators or issubclass(
                estimator_class, ComposableEstimatorMixin
            ):
                continue

            estimator = estimator_class._create_test_instance(
                parameter_set="results_comparison", return_first=True
            )
            tags = estimator.get_tags()

            if tags["non_deterministic"]:
                continue

            if tags["capability:univariate"]:
                print(estimator_name + " univariate")  # noqa: T201
                u_est = _clone_estimator(estimator, 42)
                u_est.fit(
                    deepcopy(univariate_datasets[estimator_type][0][0]),
                    deepcopy(univariate_datasets[estimator_type][0][1]),
                )
                uv_results[estimator_name] = get_expected_results(
                    u_est,
                    estimator_type,
                    deepcopy(univariate_datasets[estimator_type][1]),
                )

            if tags["capability:multivariate"]:
                print(estimator_name + " multivariate")  # noqa: T201
                m_est = _clone_estimator(estimator, 42)
                m_est.fit(
                    deepcopy(multivariate_datasets[estimator_type][0][0]),
                    deepcopy(multivariate_datasets[estimator_type][0][1]),
                )
                mv_results[estimator_name] = get_expected_results(
                    m_est,
                    estimator_type,
                    deepcopy(multivariate_datasets[estimator_type][1]),
                )

        with open(write_path, "w") as f:
            f.write(f'"""Expected results file for {estimator_type} estimators.\n\n')
            f.write("This file is automatically generated by\n")
            f.write(
                "aeon/testing/expected_results/_results_reproduction/"
                "write_expected_results.py\n"
            )
            f.write('"""\n\n')

            f.write("univariate_expected_results = {\n")
            for name, results in uv_results.items():
                f.write(f'    "{name}": {get_results_string(results, estimator_type)}')
            f.write("}\n")

            f.write("multivariate_expected_results = {\n")
            for name, results in mv_results.items():
                f.write(f'    "{name}": {get_results_string(results, estimator_type)}')
            f.write("}\n")
