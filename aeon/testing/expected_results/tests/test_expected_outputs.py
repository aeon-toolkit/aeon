"""Test expected outputs for estimators."""

from aeon.testing.expected_results.expected_classifier_results import (
    multivariate_expected_results as clf_multi,
)
from aeon.testing.expected_results.expected_classifier_results import (
    univariate_expected_results as clf_uni,
)
from aeon.testing.expected_results.expected_early_classifier_results import (
    multivariate_expected_results as early_clf_multi,
)
from aeon.testing.expected_results.expected_early_classifier_results import (
    univariate_expected_results as early_clf_uni,
)
from aeon.testing.expected_results.expected_regressor_results import (
    multivariate_expected_results as reg_multi,
)
from aeon.testing.expected_results.expected_regressor_results import (
    univariate_expected_results as reg_uni,
)
from aeon.utils.discovery import all_estimators


def test_expected_outputs():
    """Test estimators in the expected outputs dictionaries."""
    estimators = all_estimators(
        type_filter=["classifier", "early_classifier", "regressor"]
    )
    estimator_names = [e[0] for e in estimators]

    for key, value in clf_uni.items():
        assert key in estimator_names
        assert isinstance(value, list)
    for key, value in clf_multi.items():
        assert key in estimator_names
        assert isinstance(value, list)
    for key, value in early_clf_uni.items():
        assert key in estimator_names
        assert isinstance(value, list)
    for key, value in early_clf_multi.items():
        assert key in estimator_names
        assert isinstance(value, list)
    for key, value in reg_uni.items():
        assert key in estimator_names
        assert isinstance(value, list)
    for key, value in reg_multi.items():
        assert key in estimator_names
        assert isinstance(value, list)
